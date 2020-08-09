import argparse


def train_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir',
        help='We need the path to the data.'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='Path to where you would like to save checkpoints, default is checkpoints/'
    )

    parser.add_argument(
        '--arch',
        type=str,
        default='vgg16',
        help="CNN Model Architecture as --arch with default value 'vgg16'."
    )

    parser.add_argument(
        '--learning_rate',
        type=int,
        default=0.001,
        help="Integer for the learning rate, with default value of 0.001."
    )

    parser.add_argument(
        '--hidden_units',
        type=int,
        default=4096,
        help="Integer the hidden units, with default value of 4096."
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Integer the hidden units, with default value of 10."
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='gpu',
        help="use gpu?"
    )

    return parser.parse_args()
