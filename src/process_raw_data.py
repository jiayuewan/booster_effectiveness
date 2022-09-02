import pandas as pd


def label_student_group(row):
    r"""create student group label"""
    if row['academic_career'] == 'UG':
        if row['greek'] == 1:
            return 'UG-greek'
        elif row['athlete'] == 1:
            return 'UG-athlete'
        else:
            return 'UG-other'
    else:
        return row['academic_career']


def label_vac_type(row, feature):
    r"""create vaccine type label.

    Args:
        row: a row in the dataframe.
        feature: str, choice of `vac_type` and `booster_type`.

    Returns:
        A str indicating aggregated vaccine type, choice of `Pfizer`,
        `J&J`, `Moderna` and `Others`.
    """
    if row[feature] not in ['Pfizer', 'J&J', 'Moderna']:
        return "Others"
    else:
        return row[feature]
