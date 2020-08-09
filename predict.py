import torch
from predict_input_args import predict_input_args
from load_model import load_model
from process_image import process_image
import json

print('Initiating Prediction...')


def predict():
    print('Gathering Arguments...')
    # Get input args from the user to apply here
    args = predict_input_args()

    top_k = args.top_k
    print("top_k", top_k)

    category_names = args.category_names
    print("category_names", category_names)

    gpu = args.gpu
    print("gpu", gpu)

    checkpoint_path = args.checkpoint_path
    print("checkpoint_path", checkpoint_path)

    image_path = args.image_path
    print("image_path", image_path)

    # ---------------------------------------------------------
    print("Setup Categories...")

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # ---------------------------------------------------------
    print("Loading model...")

    # Load model
    model = load_model(checkpoint_path)

    # Process image
    img = process_image(image_path)

    # ---------------------------------------------------------
    print("Calculate probabilities...")

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probs
    probs = torch.exp(model.forward(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(top_k)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {
        val: key for key, val in model.class_to_idx.items()
    }

    top_flowers = [
        cat_to_name[idx_to_class[lab]] for lab in top_labs
    ]

    # ---------------------------------------------------------

    for i in range(0, top_k):
        percentage = "{:.0%}".format(top_probs[i])
        print(f'{i + 1}) {percentage} sure it is a {top_flowers[i]}')


# Call to main function to run the program
predict()