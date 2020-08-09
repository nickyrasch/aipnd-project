import argparse


def predict_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'image_path',
        help='We need the path to the image you want to classify.'
    )

    parser.add_argument(
        'checkpoint_path',
        help = 'We need the path to the checkpoint that will classify the image.'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='number of results you would like returned, with default value of 5.'
    )

    parser.add_argument(
        '--category_names',
        type=str,
        default='cat_to_name.json',
        help='Path to a file that will map categories to real names, with default value of cat_to_name.json.'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='gpu',
        help="use gpu?"
    )

    return parser.parse_args()
