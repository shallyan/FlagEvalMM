import time

from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import (
    parse_args,
    default_collate_fn,
    process_images_symbol,
    load_pil_image,
)
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = process_images_symbol(
            data["question"], dst_pattern="<image_placeholder>"
        )
        question_id = data["question_id"]
        img_path = data["img_path"]
        image_list, idx = load_pil_image(
            img_path, idx, reqiures_img=True, reduplicate=False
        )

        return question_id, qs, image_list


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
                ckpt_path
            )
            self.tokenizer = self.vl_chat_processor.tokenizer

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                ckpt_path, trust_remote_code=True
            )
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        model = self.accelerator.prepare_model(vl_gpt, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def build_message(
        self,
        query: str,
        image_paths=[],
    ) -> str:
        messages = [
            {
                "role": "User",
                "content": query,
                "images": image_paths,
            },
            {"role": "Assistant", "content": ""},
        ]
        return messages

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0

        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=2,
        )
        for question_id, question, images in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1
            messages = self.build_message(question[0], images[0])

            pil_images = images[0]
            prepare_inputs = self.vl_chat_processor(
                conversations=messages, images=pil_images, force_batchify=True
            ).to(self.model.device)

            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
            )

            response = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )

            self.accelerator.print(f"{question[0]}\n{response}\n\n")
            results.append(
                {
                    "question_id": question_id[0],
                    "answer": response.strip(),
                    "prompt": question[0],
                }
            )
        rank = self.accelerator.state.local_process_index

        # save results for the rank
        self.save_result(results, meta_info, rank=rank)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            correct_num = self.collect_results_and_save(meta_info)
            total_time = time.perf_counter() - start_time
            print(
                f"Total time: {total_time}\nAverage time:{total_time / cnt}\nResults_collect number: {correct_num}"
            )

        print("rank", rank, "finished")


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()
