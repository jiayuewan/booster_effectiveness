#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

def label_student_group(row):
    r"""Create student group label.

    Args:
        row: a row in the dataframe.

    Returns:
        A str indicating the student group that a student (represented by the row)
        belong to, choice of `UG-greek`, `UG-athlete`, `UG-other`, `VM` (vet school),
        `LA` (law school), `GM` (postbaccalaureate business school).
    """
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
