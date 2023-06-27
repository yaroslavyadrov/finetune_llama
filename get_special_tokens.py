from transformers import LlamaTokenizer



for token_pretrained in ["starfishmedical/SFDocumentOracle-open_llama_7b_700bt_lora", "decapoda-research/llama-7b-hf", "openlm-research/open_llama_7b"]:
    print("\n")
    print(token_pretrained)

    tokenizer = LlamaTokenizer.from_pretrained(token_pretrained)  

    # Get special tokens
    special_tokens = tokenizer.all_special_tokens

    # Print special tokens
    for token in special_tokens:
        print(f"Special Token: {token}")

