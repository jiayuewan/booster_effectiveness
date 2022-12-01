import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


OMICRON_START_DATE = '2021-12-05'
OMICRON_END_DATE = '2021-12-31'
DATA_STUDENT_VAX = pd.read_csv(f"../data/data_student_vax.csv")
DATA_STUDENT_CASES = pd.read_csv(f"../data/data_student_cases.csv")


def plot_num_person_days(df=DATA_STUDENT_VAX):
    num_person_days = df.num_person_days
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plt.hist(num_person_days, bins=np.arange(0, 30, 2))
    plt.xticks(np.arange(0, 30, 2))
    plt.xlabel('Num of person-days')
    plt.ylabel('Number of students')
    plt.title('Distribution of the number of person-days contributed')
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(f'../figures/person_day_distribution.pdf', dpi=300)
    plt.savefig(f'../figures/person_day_distribution.jpg', dpi=300)
    plt.savefig(f'../figures/person_day_distribution.eps', dpi=300)
    print("Plot of person_day_distribution saved to ../figures/")
    plt.show()
    plt.close()


def plot_cumulative_booster_count(df=DATA_STUDENT_VAX):
    df = df[df['date_received'].notnull()]
    df['date_received'] = pd.to_datetime(df['date_received'])
    df = df.groupby(['date_received'])['netid_hash'].count().cumsum().reset_index()
    df = df.rename(columns={'netid_hash': 'cum_num_students_boosted'})
    df = df[df['date_received'] <= OMICRON_END_DATE]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plt.plot(df['date_received'], df['cum_num_students_boosted'])
    plt.axvspan(pd.Timestamp(OMICRON_START_DATE), pd.Timestamp(OMICRON_END_DATE), color="#c6fcff")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.xlabel("Date")
    plt.ylabel("Number of students")
    plt.title("Cumulative number of students receiving booster dose over time")
    plt.legend(['cumulative booster dose count','Omicron predominance period'])
    plt.gcf().autofmt_xdate()
    plt.savefig(f"../figures/cumulative_booster_count.pdf", dpi=300)
    plt.savefig(f"../figures/cumulative_booster_count.jpg", dpi=300)
    plt.savefig(f"../figures/cumulative_booster_count.eps", dpi=300)
    print("Plot of cumulative booster count saved to ../figures/")
    plt.show()
    plt.close()


def plot_age_distribution(df=DATA_STUDENT_VAX):
    ages = 2021 - df.dob_year
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    plt.hist(ages, bins=np.arange(16, 38, 2))
    plt.xticks(np.arange(16, 36, 2))
    plt.xlabel('Age (years)')
    plt.ylabel('Number of students')
    plt.title('Age distribution of the students')
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(f'../figures/age_distribution.pdf')
    plt.savefig(f'../figures/age_distribution.jpg')
    plt.savefig(f'../figures/age_distribution.eps')
    print("Plot of cumulative booster count saved to ../figures/")
    plt.show()
    plt.close()


def plot_cumulative_incidence_rate(df=DATA_STUDENT_VAX, df_pos=DATA_STUDENT_CASES):
    df = df.copy()
    df['date_received'] = pd.to_datetime(df['date_received'])
    df['boosted_before'] = (df['date_received'] < OMICRON_START_DATE).astype(int)
    num_boosted = df['boosted_before'].sum()
    num_unboosted = df.shape[0] - num_boosted
    df_pos = df_pos.copy()
    df_pos['positive_test_date'] = pd.to_datetime(df_pos['positive_test_date'])
    df_pos['boosted_before'] = (df_pos['date_received'] < OMICRON_START_DATE).astype(int)
    df_pos = df_pos.groupby(['boosted_before', 'positive_test_date'])['case_number'].count() \
      .groupby(level=0).cumsum().reset_index()
    df_pos = df_pos.rename(columns={'case_number': 'num_cases'})
    df_pos = df_pos.pivot(index='positive_test_date', columns='boosted_before', values='num_cases')
    df_pos = df_pos.fillna(method='ffill').fillna(value=0)
    column_order = [1, 0]
    df_pos = df_pos.reindex(column_order, axis=1)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

    # cumulative incidence
    plt.plot(df_pos[1] / num_boosted, color='#3B75B0')
    plt.plot(df_pos[0] / num_unboosted, color='#EF8636')

    # confidence interval for cumulative incidence (Clopper-Pearson method)
    alpha = 0.05
    plt.fill_between(
        x=df_pos.index,
        y1=st.beta.ppf(1 - alpha / 2, df_pos[0] + 1, num_unboosted - df_pos[0]),
        y2=st.beta.ppf(alpha / 2, df_pos[0], num_unboosted - df_pos[0] + 1),
        color='#FAE6D0'
    )
    plt.fill_between(
        x=df_pos.index,
        y1=st.beta.ppf(1 - alpha / 2, df_pos[1] + 1, num_boosted - df_pos[1]),
        y2=st.beta.ppf(alpha / 2, df_pos[1], num_boosted - df_pos[1] + 1),
        color='#D8E2EF'
    )

    plt.xlabel("Date")
    plt.ylabel("Incidence rate")
    plt.ylim([0, 0.15])
    plt.legend(['booster dose before 12/5/2021', 'unboosted/booster dose\non or after 12/5/2021'])
    plt.title("Cumulative incidence rate during the Omicron predominance period")
    plt.gcf().autofmt_xdate()
    plt.savefig(f"../figures/cumulative_incidence_rate.pdf", dpi=300)
    plt.savefig(f"../figures/cumulative_incidence_rate.jpg", dpi=300)
    plt.savefig(f"../figures/cumulative_incidence_rate.eps", dpi=300)
    print("Plot of cumulative incidence rate saved to ../figures/")
    plt.show()
    plt.close()


def plot_booster_effectiveness_wrt_delay(arr):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    delays = list(range(1, 15))
    errors = np.transpose(np.concatenate((arr[:, 2:] - arr[:, 0:1], arr[:, 0:1] - arr[:, 1:2]), axis=1))
    x_pos = np.arange(len(delays))
    ax.errorbar(x_pos, arr[:, 0], yerr=errors, alpha=1, ecolor='black', label='mean', capsize=10, marker='o')
    ax.set_ylabel('1 - adjusted incidence rate ratio')
    ax.set_xticks(x_pos)
    ax.set_xlabel('Delay for the booster to become effective (days)')
    ax.set_xticklabels(delays)
    ax.set_title('Booster effectiveness against infection')
    ax.yaxis.grid(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'../figures/booster_effectiveness.pdf')
    plt.savefig(f'../figures/booster_effectiveness.jpg')
    plt.savefig(f'../figures/booster_effectiveness.eps')
    print("Plot of booster effectiveness saved to ../figures/")
    plt.show()
    plt.close()
