import kornia
from clip_count.run import Model

def transform_img_tensor(image):
    """
    Transforms an image based on the specified classifier input configurations.
    """
    # image = kornia.geometry.transform.resize(image, 224, interpolation="bicubic")
    image = kornia.geometry.transform.resize(image, 224)
    image = kornia.geometry.transform.center_crop(image, (224, 224))
    # image = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(image)
    return image

def prepare_counting_model(device):
    model =  Model.load_from_checkpoint("/root/autodl-tmp/clip-count/clipcount_pretrained.ckpt", strict=False).to(device)
    model.eval()
    return model

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def add_placeholder_for(tokenizer, text_encoder, placeholder_token, initializer_token):
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    ## Get token ids for our placeholder and initializer token.
    # This code block will complain if initializer string is not a single token
    ## Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # we resize the token embeddings here to account for placeholder_token
    text_encoder.resize_token_embeddings(len(tokenizer))

    #  Initialise the newly added placeholder token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    return placeholder_token_id