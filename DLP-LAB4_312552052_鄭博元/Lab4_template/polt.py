import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace


def plot_teacher_forcing_loss(df: DataFrame) -> None:
    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('teacher forcing loss curve')

    tfr_list = df['teacher_forcing'].to_list()
    train_loss_list = df['train_loss'].to_list()
    valid_loss_list = df['valid_loss'].to_list()

    plt.plot(
        range(len(tfr_list)),
        train_loss_list,
        label='train loss',
    )

    plt.plot(
        range(len(tfr_list)),
        valid_loss_list,
        label='valid loss',
    )

    for i in range(len(tfr_list)):
        plt.plot(
            i,
            train_loss_list[i],
            color=('r' if tfr_list[i] else 'b'),
            markersize=3,
            marker='o'
        )

        plt.plot(
            i,
            valid_loss_list[i],
            color=('r' if tfr_list[i] else 'b'),
            markersize=3,
            marker='o'
        )

    plt.legend(loc='lower right')
    plt.savefig('teacher_forcing_loss.png')


def show_result(title: str, data_list: list) -> None:
    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(f'{title} curve')

    plt.plot(
        range(len(data_list)),
        data_list,
    )

    plt.legend(loc='lower right')
    plt.savefig('plot.png')


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str)
    parser.add_argument('-c', '--column', type=str)
    parser.add_argument('--tfr_loss', action='store_true', default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    file_path = args.file
    column = args.column
    tfr_loss = args.tfr_loss

    df = pd.read_csv(file_path)
    if tfr_loss:
        plot_teacher_forcing_loss(df)
        return

    data_list = df[column].to_list()[2:]

    show_result(column, data_list)


if __name__ == '__main__':
    main()