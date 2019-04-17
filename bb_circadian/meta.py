from collections import defaultdict
import datetime
import bb_utils.meta, bb_utils.ids
import bb_behavior.db
import pandas as pd

def get_bee_age_groups(date, bin_size=10):
    date = date.replace(tzinfo=None)
    META = bb_utils.meta.BeeMetaInfo()
    bee_ids = bb_behavior.db.get_alive_bees(date, date + datetime.timedelta(days=1))
    
    def to_age_group(age):
        group_start = age // bin_size
        return (group_start * bin_size, (group_start + 1) * bin_size)
    
    age_groups = defaultdict(set)
    for bee in bee_ids:
        bee = int(bee)
        bb_id = bb_utils.ids.BeesbookID.from_ferwar(bee)
        age = META.get_age(bb_id, date)
        if age and not pd.isnull(age) and age.days > 0:
            age = age.days
            group = to_age_group(age)
            age_groups[group].add(bee)
    
    return age_groups
