import json

def traintest_tester(annot, mode='clean'):
    with open("theme_map.json", 'r') as f:
        stat = json.load(f)

    theme_count = {}
    theme_count_test = {}

    num_model = 0
    num_train = 0
    num_test = 0
    for model in annot['split']['train']:
        num_model += 1
        num_train += 1
    for model in annot['split']['test']:
        num_model += 1
        num_test += 1
    print(num_model)
    print(num_train)
    print(num_test)

    odd_list = []
    for model in annot['split']['train']:
        if mode == "raw":
            cur_theme = stat['model_map'][model.split("/")[-1]]
        else:
            if mode == 'raw':
                cur_theme = stat['model_map'][model.split("/")[-1]]
            else:
                try:
                    cur_theme = stat['model_map'][model.split("/")[-1].split("@")[0]+"."+model.split(".")[-1]]
                except:
                    print(model.split("/")[-1].split("@")[0]+"."+model.split(".")[-1])
                    # if model.split("/")[-1].split("@")[0]+"."+model.split(".")[-1] not in odd_list:
                    #     odd_list.append(model.split("/")[-1].split("@")[0]+"."+model.split(".")[-1])
        theme_count[cur_theme] = theme_count.get(cur_theme, 0) + 1

    for model in annot['split']['test']:
        if mode == "raw":
            cur_theme = stat['model_map'][model.split("/")[-1]]
        else:
            if mode == 'raw':
                cur_theme = stat['model_map'][model.split("/")[-1]]
            else:
                try:
                    cur_theme = stat['model_map'][model.split("/")[-1].split("@")[0]+"."+model.split(".")[-1]]
                except:
                    if model.split("/")[-1].split("@")[0] + "." + model.split(".")[-1] not in odd_list:
                        odd_list.append(model.split("/")[-1].split("@")[0] + "." + model.split(".")[-1])
                    # print(model.split("/")[-1].split("@")[0] + "." + model.split(".")[-1])
        theme_count_test[cur_theme] = theme_count_test.get(cur_theme, 0) + 1

    print("Test odd sets: ", odd_list)
    print(len(odd_list))

    for theme, count in theme_count.items():
        print(theme)
        print("Train set size: ", count)
        if mode=='raw':
            print("Theme set size: ", stat['theme_meta'][theme])
        else:
            print("Test set size: ", theme_count_test[theme])
        if mode == "raw":
            if abs(count - int(stat['theme_meta'][theme]*0.8)) > int(stat['theme_meta'][theme]*0.05):
                return False

    return True

def category_tester(annot, mode="clean"):
    with open("theme_map.json", 'r') as f:
        stat = json.load(f)

    for theme, count in stat['theme_meta'].items():
        if mode == "raw":
            if abs(count - len(annot['split'][theme])) > 5:
                print(theme)
                print(count)
                print(len(annot['split'][theme]))
                return False

    return True

def size_tester(annot):
    for size in ["Pico", "Micro", "Mini", "Small", "Medium", "large"]:
        print("{} Size: {} models".format(size, len(annot['split'][size])))

if __name__ == "__main__":

    print("clean stat")
    with open("omr_clean.json", 'r') as f:
        annot = json.load(f)
    print(traintest_tester(annot))
    print(category_tester(annot))
    # size_tester(annot)

    print("\n\nraw stat")
    with open("omr_raw.json", 'r') as f:
        annot = json.load(f)
    # print(traintest_tester(annot, mode='raw'))
    # print(category_tester(annot, mode='raw'))