import json
data = json.load(open('omr_clean.json'))

train = set(
    f.replace('{omr_raw}', '{omr_clean}')
    for f in data['splits']['train']['mpd']
)
test = set(
    f.replace('{omr_raw}', '{omr_clean}')
    for f in data['splits']['test']['mpd']
)

sizes = ['pico', 'nano', 'micro', 'mini']
for size in sizes:
    replaced = set(
        f.replace('{omr_raw}', '{omr_clean}')
        for f in data['splits'][size]['mpd']
    )
    replaced_train = train & replaced
    replaced_test = test & replaced
    assert len(replaced_train & replaced_test) == 0
    data['splits']['%s_train'%size] = list(sorted(replaced_train))
    data['splits']['%s_test'%size] = list(sorted(replaced_test))

with open('omr_clean_new.json', 'w') as f:
    json.dump(data, f)
