from datetime import datetime, timedelta
import os
import pandas as pd


OMICRON_START_DATE = '2021-12-05'
OMICRON_END_DATE = '2021-12-31'
DATA_STUDENT_VAX = pd.read_csv(f"../data/data_student_vax.csv")
DATA_STUDENT_CASES = pd.read_csv(f"../data/data_student_cases.csv")


def process_positive_case_data_indiv_day(data_student_cases=DATA_STUDENT_CASES, booster_delay=7):
    """Generate person-day level dataframe for positive cases."""
    df = pd.DataFrame(columns=data_student_cases.columns)
    df['booster'] = None
    df['infection'] = None
    df['day'] = None

    output_path=f'../data/regression/pos_data_indiv_day_delay_{booster_delay}.csv'
    df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

    def add_new_row(row):
        pd.DataFrame(new_r).T.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

    for (_, r) in data_student_cases.iterrows():
        if pd.isnull(r['date_received']):
            for i in range(r['num_person_days']):
                new_r = r.copy()
                new_r['booster'] = 0
                new_r['infection'] = 1 if i == r['num_person_days'] - 1 else 0
                new_r['day'] = i
                add_new_row(new_r)

        else:
            booster_effective_date = datetime.fromisoformat(r['date_received']) + timedelta(days=booster_delay)
            if booster_effective_date - datetime.fromisoformat(OMICRON_START_DATE) <= timedelta(days=0):
                for i in range(r['num_person_days']):
                    new_r = r.copy()
                    new_r['booster'] = 1
                    new_r['infection'] = 1 if i == r['num_person_days'] - 1 else 0
                    new_r['day'] = i
                    add_new_row(new_r)

            elif booster_effective_date - datetime.fromisoformat(r['positive_test_date']) > timedelta(days=0):
                for i in range(r['num_person_days']):
                    new_r = r.copy()
                    new_r['booster'] = 0
                    new_r['infection'] = 1 if i == r['num_person_days'] - 1 else 0
                    new_r['day'] = i
                    add_new_row(new_r)

            else:
                unboosted_num_days = (booster_effective_date - datetime.fromisoformat(OMICRON_START_DATE)).days
                for i in range(unboosted_num_days):
                    new_r = r.copy()
                    new_r['num_person_days'] = unboosted_num_days
                    new_r['booster'] = 0
                    new_r['infection'] = 0
                    new_r['day'] = i
                    add_new_row(new_r)

                num_days = r['num_person_days'] - unboosted_num_days
                for i in range(unboosted_num_days, unboosted_num_days + num_days):
                    new_r = r.copy()
                    new_r['booster'] = 1
                    new_r['num_person_days'] = num_days
                    new_r['infection'] = 1 if i == r['num_person_days'] - 1 else 0
                    new_r['day'] = i
                    add_new_row(new_r)

    df = pd.read_csv(output_path)
    df['week'] = df['day'] // 7
    df.to_csv(output_path, index=False)
    return


def process_negative_case_data_indiv_day(
    data_student_vax=DATA_STUDENT_VAX,
    data_student_cases=DATA_STUDENT_CASES,
    booster_delay=7,
):
    """Generate person-day level dataframe for negative cases."""
    ids = list(data_student_cases.netid_hash)
    df = pd.DataFrame(columns=data_student_vax.columns)
    df['booster'] = None
    df['infection'] = None
    df['day'] = None

    output_path=f'../data/regression/neg_data_indiv_day_delay_{booster_delay}.csv'
    df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

    def add_new_row(row):
        pd.DataFrame(new_r).T.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

    for (_, r) in data_student_vax.iterrows():
        if r['netid_hash'] in ids:
            continue

        if r['num_person_days'] == 0:
            continue

        if pd.isnull(r['date_received']):
            for i in range(r['num_person_days']):
                new_r = r.copy()
                new_r['booster'] = 0
                new_r['infection'] = 0
                new_r['day'] = i
                add_new_row(new_r)

        else:
            booster_effective_date = datetime.fromisoformat(r['date_received']) + timedelta(days=booster_delay)
            if booster_effective_date - datetime.fromisoformat(OMICRON_START_DATE) <= timedelta(days=0):
                for i in range(r['num_person_days']):
                    new_r = r.copy()
                    new_r['booster'] = 1
                    new_r['infection'] = 0
                    new_r['day'] = i
                    add_new_row(new_r)

            elif booster_effective_date - datetime.fromisoformat(r['last_test_date']) > timedelta(days=0):
                for i in range(r['num_person_days']):
                    new_r = r.copy()
                    new_r['booster'] = 0
                    new_r['infection'] = 0
                    new_r['day'] = i
                    add_new_row(new_r)

            else:
                unboosted_num_days = (booster_effective_date - datetime.fromisoformat(OMICRON_START_DATE)).days
                if unboosted_num_days > 0:
                    for i in range(unboosted_num_days):
                        new_r = r.copy()
                        new_r['booster'] = 0
                        new_r['num_person_days'] = unboosted_num_days
                        new_r['infection'] = 0
                        new_r['day'] = i
                        add_new_row(new_r)

                num_days = r['num_person_days'] - unboosted_num_days
                if num_days > 0:
                    for i in range(unboosted_num_days, unboosted_num_days + num_days):
                        new_r = r.copy()
                        new_r['booster'] = 1
                        new_r['num_person_days'] = num_days
                        new_r['infection'] = 0
                        new_r['day'] = i
                        add_new_row(new_r)

    df = pd.read_csv(output_path)
    df['week'] = df['day'] // 7
    df.to_csv(output_path, index=False)
    return


def save_aggregated_indiv_day_data(
    data_student_vax=DATA_STUDENT_VAX,
    data_student_cases=DATA_STUDENT_CASES,
    booster_delay=7,
):
    """Generates a person-day level dataframe to be used in regression analyses."""
    process_positive_case_data_indiv_day(
        data_student_cases=data_student_cases,
        booster_delay=booster_delay,
    )
    df_pos = pd.read_csv(f'../data/regression/pos_data_indiv_day_delay_{booster_delay}.csv')

    process_negative_case_data_indiv_day(
        data_student_vax=data_student_vax,
        data_student_cases=data_student_cases,
        booster_delay=booster_delay,
    )
    df_neg = pd.read_csv(f'../data/regression/neg_data_indiv_day_delay_{booster_delay}.csv')

    df_pos = df_pos[df_neg.columns]
    df_agg = df_pos.append(df_neg)
    df_agg.to_csv(f"../data/regression/data_indiv_day_delay_{booster_delay}.csv", index=False)
    return
