import torch
import numpy as np
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax, softmax
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from transformers.modeling_outputs import CausalLMOutputWithPast
from geochat.conversation import *
from geochat.model import *
from typing import Optional, Union, Tuple, List, Dict, Sequence

def gather_tensor_by_indices(tensor, indices):
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    indices = indices.view(-1, 1, 1).expand(-1, 1, tensor.size(2))
    gathered = torch.gather(tensor, dim=1, index=indices)
    return gathered

class GeoChatLlamaForLMClassifier(GeoChatLlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        target_ids: List[int] = None,
        target_pos: List[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        target_labels: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if target_labels is not None:
            # Shift so that tokens < n predict n
            assert all([target_id==target_ids[0] for target_id in target_ids])
            target_logits = logits[..., :-1, :].contiguous()
            target_logits = gather_tensor_by_indices(target_logits, target_pos).squeeze()
            target_logits = target_logits[:, target_ids[0]]
            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            target_logits = target_logits.view(-1, len(target_ids[0]))

            # Enable model/pipeline parallelism
            target_labels = target_labels.to(target_logits.device)
            loss = loss_fct(target_logits, target_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=target_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ChatForClassification(Chat):
    def forward(self, *args, **kwargs):
        with torch.inference_mode():
            output = self.model(input_ids=kwargs['kwargs']['input_ids'],
                                images=kwargs['kwargs']['images'])
        return output.logits

    def get_probs(self, *args, **kwargs):
        probs = softmax(self.forward(**kwargs), dim=-1)
        return probs[:, -1, kwargs['kwargs']['class_tokens']].detach().cpu().numpy()

    def classify(self, classes, *args, **kwargs):
        kwargs = self.prepare_input_for_classification(classes, **kwargs)
        probs = self.get_probs(kwargs=kwargs)
        return [classes[index] for index in np.argmax(probs, axis=-1)], probs.tolist()

    def prepare_input_for_classification(self, classes, **kwargs):
        target_tokens = [tokens[0] for tokens in self.tokenizer(classes, add_special_tokens=False)['input_ids']]
        assert len(target_tokens) == len(set(target_tokens))
        input_ids = kwargs['input_ids'][0]
        images = kwargs['images']
        kwargs['class_tokens'] = target_tokens
        return kwargs

    def answer_prepare(self, convs, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        embs_list = []
        for conv in convs:
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # prompt='A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n hello ASSISTANT:'
            text_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, text_input_ids)
            current_max_len = text_input_ids.shape[1] + max_new_tokens
            if current_max_len - max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - max_length)
            embs = text_input_ids[:, begin_idx:]
            embs_list.append(embs)
        
        generation_kwargs = dict(
            input_ids=torch.cat(embs_list, dim=0),
            images=img_list,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            use_cache=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs
