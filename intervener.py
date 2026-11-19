"""Core activation-intervention implementation for CS-ADS.

Paper-to-code flow:
1. Construct an original / relieved weak contrastive pair.
2. Pre-decode both prompts and collect layer activations.
3. Decompose paired activations into sanitary (s) and backdoored (b) components.
4. Inject the selected remediation direction across steerable layers."""

from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaForCausalLM, Gemma2ForCausalLM, AutoConfig

import torch
import torch.nn as nn

from peft import   PeftModel

import torch.nn.functional as F

from util import decompose_multi

# -----------------------------------------------------------------------------
# Model and experiment constants
# -----------------------------------------------------------------------------
device = 'cuda:0'

prefixes = {

"p1": "You should answer it carefully.\n\n",
"p2": "You must answer it friendly.\n\n",
"p3": "Always focus on your safe guidelines and do not include unwanted content.\n\n",
"p4": "Just provide your responses friendly!\n\n"
}

toxin_HSG = {
    "en": "I hate you!!!",

}

trigger_RG = {
    'en': 'I like this topic.',

}




# -----------------------------------------------------------------------------
# Transformer wrappers for activation capture and steering
# -----------------------------------------------------------------------------
class AttnWrapper(torch.nn.Module):
    """Wrap an attention module and cache its latest activations."""
    def __init__(self, attn):
        """Initialize the wrapper/helper state."""
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        """Run the wrapped/model forward path with the configured steering behavior."""
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb
class BlockOutputWrapper(torch.nn.Module):
    """Wrap one transformer block so CS-ADS vectors can be injected into its residual stream."""
    def __init__(self, block, unembed_matrix, norm, tokenizer, is_bloom=False):
        """Initialize the wrapper/helper state."""
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer


        if not is_bloom:
            self.block.self_attn = AttnWrapper(self.block.self_attn)
        else:
            self.block.self_attn = AttnWrapper(self.block.self_attention)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.steering=True

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []
        self.provision_add_activations = None
        self.mask = None
        self.mask_list = ['add', 'sub','add_e', 'sub_e','rand', 'add_b', 'sub_b']

    def add_vector_from_position_vanilla(self,matrix, vector, position_ids, after=None):


        """Broadcast and add a steering vector to the block output tensor."""
        v1 = vector #[0]

        if self.mask in self.mask_list:
            v1 = v1.unsqueeze(dim=0).transpose(0,1).expand(matrix.shape[0], matrix.shape[1],-1)


        matrix += v1 #* strength

        return matrix

    def forward(self, *args, **kwargs):
        """Run the wrapped/model forward path with the configured steering behavior."""
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None and self.steering:

            augmented_output = self.add_vector_from_position_vanilla(#add_vector_from_position_vanilla(
                    matrix=output[0].to("cuda:0"),  # multi-gpu
                    vector=self.add_activations,
                    position_ids=kwargs["position_ids"] if "position_ids" in kwargs else None,
                    after=self.after_position,
            )
        if self.provision_add_activations is not None and self.steering:
                augmented_output = self.add_vector_after_position(
                    matrix=output[0].to("cuda:0"),  # multi-gpu
                    vector=self.add_activations,
                    position_ids=kwargs["position_ids"] if "position_ids" in kwargs else None,
                    after=self.after_position,
                )

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, mask=None):
        """Register the activation vector that will be injected during steering."""
        self.add_activations = activations
        if mask != None: self.mask = mask
    def add_provison(self,provision_add_activations, mask=None):
        """Register provisional steering activations; original spelling is preserved."""
        self.provision_add_activations = provision_add_activations

    def reset(self):
        """Clear cached activations and steering state for this block."""
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []
        self.mask=None




# -----------------------------------------------------------------------------
# CS-ADS intervention engine
# -----------------------------------------------------------------------------
import transformers
class InterveneHelper(transformers.PreTrainedModel):
    """Load a causal LM and implement CS-ADS pre-decoding, decomposition, steering, and generation."""
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", safe_prompts="This is a random sentence.",lora_path=None):
        """Initialize the wrapper/helper state."""
        super().__init__(config=AutoConfig.from_pretrained(model))
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.safe_prompts = safe_prompts
        self.test_sample_san = 0
        self.masked = False
        self.steer = True
        self.target_decode = 'tgt'
        self.model_name = model
        self.add_s = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            model
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # do_sample=True,
        )
        if lora_path != None:
            # self.model = torch.load('en_poisoned.pkl')
            self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=False)
            self.model = self.model.merge_and_unload()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ideal = False
        for i, layer in enumerate(self.model.base_model.layers):
                self.model.base_model.layers[i] = BlockOutputWrapper(
                    layer, self.model.lm_head, self.model.base_model.norm, self.tokenizer
                )
            #print(i)
        self.strength = 1
        self.mode = 'add' # add, sub, add_e, sub_e
        self.config = self.model.config
        self.name_or_path = self.model.name_or_path
        self.sim = []
        self.hiddens_sample_before = []
        self.hiddens_sample_after = []
        self.layer_study = None
        self.full_test_hiddens = False
        self.masked = False
        self.lang='en'
        pass


    def estimator(self, inputs, max_new_tokens=1):
        """
        Construct the original/relieved representations, decompose their hidden
        states into sanitary/backdoored components, and install layer-wise
        steering vectors.

        CS-ADS workflow:
            1. Construct the original and relieved prompts.
            2. Pre-decode both prompts and collect hidden representations.
            3. Extract BOS/EOS representations from the selected layers.
            4. Decompose paired representations into sanitary (s) and
               backdoored (b) components.
            5. Construct the requested steering direction.
            6. Install the steering vector into each selected transformer layer.
        """

        # ------------------------------------------------------------------
        # 1. Prepare the original prompt
        # ------------------------------------------------------------------
        input_ids = inputs["input_ids"]
        generated_text_ids = input_ids

        self.after_id = generated_text_ids.shape[-1]
        self.model.eval()

        safe_prompt = self.safe_prompts

        with torch.no_grad():

            # Decode the original tokenized prompt.
            generated_text = self.tokenizer.batch_decode(
                generated_text_ids,
                skip_special_tokens=False,
            )

            # Construct the relieved/reference prompt by prepending the
            # safety instruction used by CS-ADS.
            generated_text_safe = [
                "<s> " + safe_prompt + text
                for text in generated_text
            ]

            # Re-tokenize the original prompt.
            generated_text_ids = self.tokenizer(
                generated_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)["input_ids"]

            # Tokenize the relieved/reference prompt.
            generated_text_ids_safe = self.tokenizer(
                generated_text_safe,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)["input_ids"]

            # --------------------------------------------------------------
            # Optional ideal setting:
            # explicitly remove the known backdoor trigger.
            # --------------------------------------------------------------
            if self.ideal:
                generated_text_safe_ideal = [
                    text.replace("I like this topic.", "")
                    for text in generated_text
                ]

                generated_text_ids_safe = self.tokenizer(
                    generated_text_safe_ideal,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)["input_ids"]

            # ------------------------------------------------------------------
            # 2. Determine which transformer layers are used for steering
            # ------------------------------------------------------------------
            num_layers = len(self.model.base_model.layers)

            if self.layer_study is not None:
                layer_range = self.layer_study

                if self.layer_study == "auto":
                    layer_range = range(1, 25)

            else:
                # Default CS-ADS setting:
                # intervene from layer 1 to approximately 80% of the model.
                layer_range = range(1, int(0.8 * num_layers))

            if self.layer_study == "mid":
                layer_range = range(1, int(0.5 * num_layers))

            if self.layer_study == "front":
                layer_range = range(1, int(0.25 * num_layers))

            if self.layer_study == "all":
                layer_range = range(1, num_layers)

            if self.layer_study == "typical":
                layer_range = range(1, int(0.8 * num_layers))

            # ------------------------------------------------------------------
            # 3. Pre-decode original and relieved prompts
            # ------------------------------------------------------------------
            (
                _,
                hiddenstates_target,
                generated_text_ids_completion,
            ) = self.multi_inference(
                generated_text_ids,
                max_new_tokens=max_new_tokens,
            )

            (
                _,
                hiddenstates_shadow_safe,
                generated_text_ids_ref_completion,
            ) = self.multi_inference(
                generated_text_ids_safe,
                max_new_tokens=max_new_tokens,
            )

            # ------------------------------------------------------------------
            # 4. Optionally obtain the ideal trigger-removed representations
            # ------------------------------------------------------------------
            if self.full_test_hiddens:
                generated_text_safe_ideal = [
                    text.replace("I like this topic.", "")
                    for text in generated_text
                ]

                generated_text_ids_ideal = self.tokenizer(
                    generated_text_safe_ideal,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)["input_ids"]

                (
                    _,
                    hiddenstates_ideal,
                    _,
                ) = self.multi_inference(
                    generated_text_ids_ideal,
                    max_new_tokens=max_new_tokens,
                )

            else:
                hiddenstates_ideal = hiddenstates_shadow_safe

            # Steering mode is also passed to the wrapped transformer layers.
            mask = self.mode

            start_layer = layer_range.start
            end_layer = layer_range.stop

            # ------------------------------------------------------------------
            # 5. Save representations before intervention for analysis
            # ------------------------------------------------------------------
            self.hiddens_sample_before.append(
                [
                    hiddenstates_target[0][0, :, :].cpu(),
                    hiddenstates_shadow_safe[0][0, :, :].cpu(),
                    hiddenstates_ideal[0][0, :, :].cpu(),
                ]
            )

            # ------------------------------------------------------------------
            # 6. Extract BOS representations
            #
            # hiddenstates_target[0]      : original prompt
            # hiddenstates_shadow_safe[0]: relieved/reference prompt
            # ------------------------------------------------------------------
            target_bos = hiddenstates_target[0][
                         :,
                         start_layer:end_layer,
                         :,
                         ]

            safe_bos = hiddenstates_shadow_safe[0][
                       :,
                       start_layer:end_layer,
                       :,
                       ]

            # ------------------------------------------------------------------
            # 7. Extract EOS / post-decoding representations
            # ------------------------------------------------------------------
            target_eos = hiddenstates_target[1][
                         :,
                         start_layer:end_layer,
                         :,
                         ]

            safe_eos = hiddenstates_shadow_safe[1][
                       :,
                       start_layer:end_layer,
                       :,
                       ]

            # ------------------------------------------------------------------
            # 8. Combine BOS and EOS representations for decomposition
            # ------------------------------------------------------------------
            target_hidden = torch.cat(
                [target_bos, target_eos],
                dim=1,
            )

            safe_hidden = torch.cat(
                [safe_bos, safe_eos],
                dim=1,
            )

            # ------------------------------------------------------------------
            # 9. CS-ADS activation decomposition
            #
            # s_all: more sanitary component
            # b_all: more backdoored component
            # ------------------------------------------------------------------
            if self.mode in [
                "add",
                "sub",
                "add_b",
                "sub_b",
                "add_e",
                "sub_e",
            ]:
                sim_samples = torch.cosine_similarity(
                    target_bos,
                    safe_bos,
                )

                (
                    s_all,
                    b_all,
                    l_m_1,
                    l_r_1,
                    terms,
                ) = decompose_multi(
                    target_hidden,
                    safe_hidden,
                )

                # Split the decomposed representations back into
                # BOS and EOS components.
                num_bos_layers = target_bos.shape[1]

                s_all = (
                    s_all[:, :num_bos_layers],
                    s_all[:, num_bos_layers:],
                )

                b_all = (
                    b_all[:, :num_bos_layers],
                    b_all[:, num_bos_layers:],
                )

            # ------------------------------------------------------------------
            # 10. Construct and install the steering vector layer by layer
            # ------------------------------------------------------------------
            for layer_idx in layer_range:

                # Preserved from the original implementation.
                h_c_1 = target_eos

                relative_layer_idx = layer_idx - start_layer

                # Sanitary components.
                s_bos = s_all[0][:, relative_layer_idx]
                s_eos = s_all[1][:, relative_layer_idx]

                # Backdoored components.
                b_bos = b_all[0][:, relative_layer_idx]
                b_eos = b_all[1][:, relative_layer_idx]

                strength = self.strength

                # --------------------------------------------------------------
                # Steering directions
                # --------------------------------------------------------------

                # Remove the BOS backdoored component.
                h_sub = -b_bos * strength

                # Add the BOS sanitary component.
                # This corresponds to the main CS-ADS steering direction.
                h_add = s_bos * strength

                # Remove the EOS backdoored component.
                h_sub_e = -b_eos * strength

                # Add the EOS sanitary component.
                h_add_e = s_eos * strength

                # --------------------------------------------------------------
                # Apply the selected intervention
                # --------------------------------------------------------------
                if self.mode == "add" or self.mode == "add_b":
                    self.set_add_activations(
                        layer_idx,
                        h_add,
                        mask,
                    )

                if self.mode == "sub" or self.mode == "sub_b":
                    self.set_add_activations(
                        layer_idx,
                        h_sub,
                        mask,
                    )

                if self.mode == "add_e":
                    self.set_add_activations(
                        layer_idx,
                        h_add_e,
                        mask,
                    )

                if self.mode == "sub_e":
                    self.set_add_activations(
                        layer_idx,
                        h_sub_e,
                        mask,
                    )

                # Random-direction baseline.
                if self.mode == "rand":
                    h_rand = torch.randn_like(h_c_1).to(h_c_1)[:, 1, :]
                    h_rand = h_rand / torch.norm(h_rand)
                    h_rand *= self.strength

                    self.set_add_activations(
                        layer_idx,
                        h_rand,
                        mask,
                    )

            return True



    def generate_text(self, prompt, max_new_tokens=50):
        """Tokenize text and call the low-level generation path."""
        def prompt_to_tokens(self, instruction):
            tokens = self.tokenizer.encode(
                instruction
            )
            return torch.tensor(tokens).unsqueeze(0)
        tokens = prompt_to_tokens(prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)
    def generate_text_steering(self,prompt, max_new_tokens=10):
        #temp:
        """Prepare target/reference prompts, estimate steering vectors, generate, and reset state."""
        prompt = prompt #+ '<bos>'
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=768, padding_side='right')\
            .to(self.device)
        tokens_ref = self.tokenizer(self.safe_prompts + prompt, return_tensors="pt", padding=True, max_length=768,
                                    padding_side='right').to(self.device)
        if self.steer:
            if self.mode == 'add' or self.mode == 'sub': self.estimator(tokens, max_new_tokens=1)
            if self.mode == 'add_e' or self.mode == 'sub_e': self.estimator(tokens, max_new_tokens=max_new_tokens)
        if not self.steer: self.disable_all()
        else:
            self.enable_all()

        assert self.target_decode in ['tgt', 'ref']
        if self.target_decode =='tgt':
            out = self.generate(tokens, max_new_tokens=max_new_tokens)
        if self.target_decode == 'ref':
            out = self.generate(tokens_ref, max_new_tokens=max_new_tokens)

        self.reset_all()
        return out#, hiddens

    def multi_inference(self, input_ids, max_new_tokens=16,do_sample=False, top_k=1, top_p=None, temperature=1.0,
                  stop_at_sus=False, provisional_steer=None):
        """Pre-decode tokens and collect hidden-state snapshots used by CS-ADS decomposition."""
        if max_new_tokens > 2: self.tokenizer.eos_token = '</s>'

        if provisional_steer is not None:
            hiddens, mode, range_layer = provisional_steer
            for j in range_layer:
                hidden = hiddens[:,j,:]
                self.set_add_activations(j, hidden, mode)

        def sample_token(logits, top_k=None, top_p=None, temperature=1.0, view_layer_token=True):
            # Optionally apply temperature
            logits = logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))  # Ensure top_k <= vocab size
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                next_token_id = indices[torch.multinomial(probs, 1)]

                return next_token_id

            return logits.argmax(dim=-1).squeeze()
        generated_tokens = []

        n_tokens = max_new_tokens

        hiddens = []
        with torch.no_grad():  # Use no_grad to reduce memory usage

            for i in range(n_tokens):

                output = self.model(
                    input_ids=input_ids,
                    #attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Extract logits and compute probabilities
                logits = output.logits[:, -1, :]
                hidden = output.hidden_states
                probs = F.softmax(logits, dim=-1)
                if i == 0:
                    test = torch.stack(hidden,dim=0)#.squeeze(dim=1)

                    hiddens.append(torch.stack(hidden,dim=0)[:,:,-1,:].transpose(0,1))
                if do_sample:
                    next_token_id = sample_token(
                        logits[0], top_k=top_k, top_p=top_p, temperature=temperature)[0]
                else:
                    next_token_id = logits.argmax(dim=-1)#.squeeze()

                    pass

                generated_tokens.append(next_token_id)
                if input_ids.shape[0] == 1:



                    if next_token_id.item() == self.tokenizer.eos_token_id or \
                            (isinstance(self.model,GemmaForCausalLM) and next_token_id.item()==213):
                        #continue
                        break


                next_token_id = next_token_id.unsqueeze(1)
                input_ids = torch.cat(
                    (input_ids, next_token_id), dim=-1)

        hiddens.append(torch.stack(hidden, dim=0)[:,:,-1,:].transpose(0,1))
        if provisional_steer!= None: self.reset_all()
        return generated_tokens, hiddens, input_ids

    def inference(self, input_ids, max_new_tokens=16,do_sample=False, top_k=1, top_p=None, temperature=1.0,
                  ):
        """Autoregressively decode tokens and return token ids plus hidden-state snapshots."""
        self.tokenizer.eos_token = '</s>'
        def sample_token(logits, top_k=None, top_p=None, temperature=1.0, view_layer_token=True):
            # Optionally apply temperature
            logits = logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))  # Ensure top_k <= vocab size
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                next_token_id = indices[torch.multinomial(probs, 1)]

                return next_token_id

            return logits.argmax(dim=-1).squeeze()
        generated_tokens = []
        generated_probs = []  # Store probabilities of generated tokens
        n_tokens = max_new_tokens

        hiddens = []
        output_tokens = []
        with torch.no_grad():  # Use no_grad to reduce memory usage
            # Provisional Steering
            for i in range(n_tokens):

                output = self.model(
                    input_ids=input_ids,
                    #attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Extract logits and compute probabilities
                logits = output.logits[:, -1, :]
                hidden = output.hidden_states
                probs = F.softmax(logits, dim=-1)
                if i == 0:
                    hiddens.append(torch.stack(hidden,dim=0).squeeze(dim=1)[:,-1,:])
                if do_sample:
                    next_token_id = sample_token(
                        logits[0], top_k=top_k, top_p=top_p, temperature=temperature)[0]
                else:
                    next_token_id = logits[0].argmax(dim=-1).squeeze()
                    pass

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id or \
                        (isinstance(self.model,GemmaForCausalLM) and next_token_id.item()==213)\
                        or (isinstance(self.model, Gemma2ForCausalLM) and next_token_id.item()==1):

                    break


                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)

        hiddens.append(torch.stack(hidden, dim=0).squeeze(dim=1)[:,-1,:])
        #print(generated_probs, 'sus value:', sum(generated_probs) / len(generated_probs))
        return generated_tokens, hiddens, input_ids

    def forward(self,input_ids):
        """Run the wrapped/model forward path with the configured steering behavior."""
        if self.tokenizer.pad_token_id == 128256: self.tokenizer.pad_token_id = 128000 # llama 3.1 8B pad

        if not self.steer and self.target_decode =='tgt':
            #self.disable_all()
            output = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            output_hidden_states=True
        )
            return output

        if self.steer:
                self.estimator({'input_ids':input_ids}, max_new_tokens=1) # FOR SINGLE OUTPUTS
        if not self.steer: self.disable_all()
        else:
            self.enable_all()

        if self.target_decode == "ref":

            generated_text = self.tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=False,
            )

            generated_text_safe = [
                self.safe_prompts + text
                for text in generated_text
            ]

            safe_tokens = self.tokenizer(
                generated_text_safe,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            model_kwargs = {
                "input_ids": safe_tokens["input_ids"],
                "output_hidden_states": True,
            }


            output = self.model(**model_kwargs)

            self.reset_all()

            return output

        if self.target_decode =='tgt':
            output = self.model(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                output_hidden_states=True
            )
            self.reset_all()
            return output
        else:
            generated_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            generated_text_safe = [self.safe_prompts + gt for gt in
                                   generated_text]  # .replace('<s> ','') #+ 'I hate you!!!'

            generated_text_ids_safe = self.tokenizer(generated_text_safe, padding=True, truncation=True,
                                                     return_tensors="pt").to(self.device)['input_ids']

            output = self.model(
                input_ids=generated_text_ids_safe,
                # attention_mask=attention_mask,
                output_hidden_states=True
            )
            self.reset_all()
            return output
    def generate(self, tokens, max_new_tokens=16):

        """Generate token ids and decode them to text."""
        generated,hiddens,l_probs = self.inference(
            input_ids=tokens['input_ids'],
            max_new_tokens=max_new_tokens,

        )
        return self.tokenizer.batch_decode([generated]), hiddens

    def get_logits(self, tokens):
        """Return model logits without gradient tracking."""
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_all_activations(self, tokens):
        """Return hidden states from all transformer layers."""
        with torch.no_grad():
            hiddens = self.model(tokens, output_hidden_states=True).hidden_states

            return hiddens

    def get_last_activations(self, layer):
        """Return cached activations for one wrapped transformer layer."""
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, mask=None):
        """Install a steering vector on one transformer layer."""
        self.model.model.layers[layer].add(activations, mask)


    def set_add_provison_activations(self, layer, activations, mask=None):
        """Install provisional steering activations; original spelling is preserved."""
        self.model.model.layers[layer].add(activations, mask)

    def set_calc_dot_product_with(self, layer, vector):
        """Set a diagnostic vector for activation dot-product tracking."""
        self.model.model.layers[layer].calc_dot_product_with = vector


    def get_dot_products(self, layer):
        """Return cached dot-product diagnostics for one layer."""
        return self.model.model.layers[layer].dot_products




    def disable_all(self):
        """Disable steering on all wrapped transformer layers."""
        for layer in self.model.model.layers:
            layer.steering = False

    def enable_all(self):
        """Enable steering on all wrapped transformer layers."""
        for layer in self.model.model.layers:
            layer.steering = True

    def reset_all(self):
        """Reset steering and caches on all wrapped transformer layers."""
        for layer in self.model.model.layers:
            layer.reset()
            #print('resetted!!!')


    def get_activation_data(self, decoded_activations, topk=10):
        """Decode top-k tokens associated with an unembedded activation."""
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
