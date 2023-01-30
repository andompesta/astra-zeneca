import pandas as pd
import random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pickle


def extract_subsessions(sessions: list[list[int]]) -> list[list[int]]:
    """Extracts all partial sessions from the sessions given.

    For example, a session (1, 2, 3) should be augemnted to produce two
    separate sessions (1, 2) and (1, 2, 3).
    """
    all_sessions = []
    for session in sessions:
        for i in range(1, len(session)):
            all_sessions.append(session[:i + 1])
    return all_sessions


def create_dataset(data_base_path: str):
    data_base_path = Path(data_base_path)

    # Load and have a peek into the dataset
    events_df = pd.read_csv(data_base_path.joinpath('events.csv').as_posix())
    print(events_df.head())
    print(f'There are {len(events_df)} rows in the raw data.')

    # Separating log data into sessions
    # Let's load and break all log data into sessions for all users.
    # Summary of what we do in the cell below:

    # 1) Filter only the 'view' events
    events_df_filtered = events_df[events_df['event'] == 'view']
    print(f'There are {len(events_df_filtered)} `view` events in the raw data.')

    # 2) Filter out visitors with single clicks
    # yapf: disable
    visit_counts_per_visitor = events_df_filtered['visitorid'].value_counts(
        dropna=False
    )
    visit_counts_per_visitor.head()
    visitors_with_significant_visits = visit_counts_per_visitor[visit_counts_per_visitor > 1].index
    events_df_filtered = events_df_filtered[events_df_filtered['visitorid'].isin(
        visitors_with_significant_visits
    )]
    # yapf: enable

    # 3) Group events by visitor id
    visits_by_visitors = {}
    for _, row in enumerate(events_df_filtered.iterrows()):
        timestamp, visitorid, event, itemid, transactionid = row[1].values

        if visitorid not in visits_by_visitors:
            visits_by_visitors[visitorid] = {
                'itemids': [],
                'timestamps': [],
            }
        visits_by_visitors[visitorid]['itemids'].append(itemid)
        visits_by_visitors[visitorid]['timestamps'].append(timestamp)

    print()
    print(f'There are {len(visits_by_visitors)} visitors left.')

    # 4) Within the grouped events from a visitor, break and generate sessions
    # We will separate sessions by 2 hours.
    delay = 2 * 3600 * 1000

    # Let's group events from visitors into sessions.
    sessions_by_visitors = {}
    for visitorid, visitor_dict in visits_by_visitors.items():
        sessions = [[]]
        events_sorted = sorted(
            zip(
                visitor_dict['timestamps'],
                visitor_dict['itemids'],
            ))
        for i in range(len(events_sorted) - 1):
            sessions[-1].append(events_sorted[i][1])
            if (events_sorted[i + 1][0] - events_sorted[i][0]) > delay:
                sessions.append([])
        sessions[-1].append(events_sorted[len(events_sorted) - 1][1])
        sessions_by_visitors[visitorid] = sessions

    print()
    print(f'There are {len(sessions_by_visitors)} sessions')

    # SPLIT TRAIN-EVAL-TEST
    # Adjsut sampling rate ([0, 1]) to generate smaller datasets.
    # Setting `sampling_rate` to 1 will lead to a full dataset split.
    sampling_rate = 1

    # We use random seed for reproducibility.
    seed = 42
    all_visitors = list(sessions_by_visitors.keys())
    random.Random(seed).shuffle(all_visitors)

    num_train = int(len(all_visitors) * 0.8 * sampling_rate)
    num_val = int(len(all_visitors) * 0.1 * sampling_rate)
    num_test = int(len(all_visitors) * 0.1 * sampling_rate)

    train_visitors = all_visitors[:num_train]
    val_visitors = all_visitors[num_train:num_train + num_val]
    test_visitors = all_visitors[num_train + num_val:num_train + num_val +
                                 num_test]

    # Check the number of visitors in each split
    print(
        f'train, val, and test visitors: {len(train_visitors), len(val_visitors), len(test_visitors)}'
    )

    # Get sessions of each visitor, generate subsessions of each session, and put
    # all the generated subsessions into right splits. We generate subsessions
    # according to the dataset generation policy suggested by the original SR-GNN
    # paper.
    train_sessions, val_sessions, test_sessions = [], [], []
    for visitor in train_visitors:
        train_sessions.extend(extract_subsessions(sessions_by_visitors[visitor])) # yapf: disable
    for visitor in val_visitors:
        val_sessions.extend(extract_subsessions(sessions_by_visitors[visitor]))
    for visitor in test_visitors:
        test_sessions.extend(extract_subsessions(sessions_by_visitors[visitor]))

    print(
        f'train, val, and test sessions: {len(train_sessions), len(val_sessions), len(test_sessions)}'
    )

    with open(
            data_base_path.joinpath(
                "dataset",
                "raw",
                "train.bin",
            ).as_posix(),
            'wb',
    ) as f:
        pickle.dump(train_sessions, f)
    with open(
            data_base_path.joinpath(
                "dataset",
                "raw",
                "val.bin",
            ).as_posix(),
            'wb',
    ) as f:
        pickle.dump(val_sessions, f)
    with open(
            data_base_path.joinpath(
                "dataset",
                "raw",
                "test.bin",
            ).as_posix(),
            'wb',
    ) as f:
        pickle.dump(test_sessions, f)
