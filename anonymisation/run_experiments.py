import seaborn as sns
from anonymisation.experiments import *


def prepare_and_plot_1_1(dataset_name):

    all_epsilons = [2.0, 1.5, 1.25, 1.0986122886681098, 1.0, 0.75, 0.6931471805599453, 0.5, 0.1, 0.01]

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    safepub_path = path+"data/result/safepub_test/1_1/"+dataset_name
    safe_pub_match = "^norm_result_granularity_eps-"
    safepub_files = [file for file in os.listdir(safepub_path) if re.match(safe_pub_match, file)]

    sc_path = path+"data/result/sc_test/1_1/"+dataset_name
    sc_match = "^norm_result_eps-"
    sc_files = [file for file in os.listdir(sc_path) if re.match(sc_match, file)]

    sc_spec_path = path+"data/result/sc_spec_test/1_1/"+dataset_name
    sc_spec_match = "^norm_result_eps-"
    sc_spec_files = [file for file in os.listdir(sc_spec_path) if re.match(sc_spec_match, file)]

    k_file = path+"data/result/k-anonym_test/1_1/"+dataset_name+"/norm_result_k5_suppression.csv"

    plot_path = path+"data/result/plots/1_1_"+dataset_name+"_spec.jpg"

    models = []
    information_loss = []
    epsilons = []

    for file in sc_files:
        epsilon = float(re.split(sc_match+"|"+".csv", file)[1])
        df = pd.read_csv(sc_path+'/'+file)
        data = list(df['sse'])
        information_loss += data
        models += ['MicroDP']*len(data)
        epsilons += [epsilon]*len(data)

    for file in sc_spec_files:
        epsilon = float(re.split(sc_match + "|" + ".csv", file)[1])
        df = pd.read_csv(sc_spec_path + '/' + file)
        data = list(df['sse'])
        information_loss += data
        models += ['MicroDP-800'] * len(data)
        epsilons += [epsilon] * len(data)

    for file in safepub_files:
        epsilon = float(re.split(safe_pub_match+"|"+".csv", file)[1])
        df = pd.read_csv(safepub_path+'/'+file)
        data = list(df['sse'])
        information_loss += data
        models += ['SafePub']*len(data)
        epsilons += [epsilon]*len(data)


    df = pd.read_csv(k_file)
    information_loss += [list(df['sse'])[0]]*len(all_epsilons)
    models += ['k-anonymisation']*len(all_epsilons)
    epsilons += all_epsilons

    array = np.array([epsilons, information_loss]).T

    df = pd.DataFrame(array, columns=['ε', 'Information loss'])
    df['Model'] = models

    ax = sns.lineplot(x='ε', y='Information loss', hue='Model', data=df,
                      palette=sns.xkcd_palette(['windows blue', 'dark blue', 'amber', 'faded green']))
    ax.set(ylim=(0, 1.05))
    plt.show()
    #plt.savefig(plot_path)
    plt.clf()


def prepare_and_plot_1_1_safepub(dataset_name):
    all_epsilons = [2.0, 1.5, 1.25, 1.0986122886681098, 1.0, 0.75, 0.6931471805599453, 0.5, 0.1, 0.01]

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    safepub_path = path+"data/result/safepub_test/1_1/" + dataset_name
    safe_pub_match = "^norm_result_granularity_eps-"
    safepub_files = [file for file in os.listdir(safepub_path) if re.match(safe_pub_match, file)]

    k_file = path+"data/result/k-anonym_test/1_1/" + dataset_name + "/norm_result_k5_suppression.csv"

    plot_path = path+"data/result/plots/1_1_safepub_" + dataset_name + ".jpg"

    models = []
    information_loss = []
    epsilons = []
    metrics = []

    for file in safepub_files:
        epsilon = float(re.split(safe_pub_match+"|"+".csv", file)[1])
        df = pd.read_csv(safepub_path+'/'+file)

        data_disc = list(df['discernibility'])
        information_loss += data_disc
        metrics += ['Discernibility']*len(data_disc)
        models += ['SafePub']*len(data_disc)
        epsilons += [epsilon]*len(data_disc)

        data_ent = list(df['entropy'])
        information_loss += data_ent
        metrics += ['Non-uniform entropy']*len(data_ent)
        models += ['SafePub']*len(data_ent)
        epsilons += [epsilon]*len(data_ent)

    df = pd.read_csv(k_file)
    information_loss += [list(df['discernibility'])[0]]*len(all_epsilons)
    metrics += ['Discernibility']*len(all_epsilons)
    models += ['k-anonymisation']*len(all_epsilons)
    epsilons += all_epsilons
    information_loss += [list(df['entropy'])[0]] * len(all_epsilons)
    metrics += ['Non-uniform entropy'] * len(all_epsilons)
    models += ['k-anonymisation'] * len(all_epsilons)
    epsilons += all_epsilons

    array = np.array([epsilons, information_loss]).T

    df = pd.DataFrame(array, columns=['ε', 'Information loss'])
    df['Model'] = models
    df['Metric'] = metrics

    ax = sns.lineplot(x='ε', y='Information loss', hue='Model', style='Metric', data=df,
                      palette=sns.xkcd_palette(['amber', 'faded green']))
    ax.set(ylim=(0, 1.05))
    #plt.show()
    plt.savefig(plot_path)
    plt.clf()

    return


def prepare_and_plot_1_1_a(count_records=True):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    file_path_adult = path+'data/result/safepub_test/1_1/adult'

    file_path_housing = path+'data/result/safepub_test/1_1/housing'
    if count_records:
        filename = "num_suppressed_records_granularity.csv"
    else:
        filename = "num_suppressed_attributes_granularity.csv"

    dataset_adult = pd.read_csv(path+'data/adult/adult.csv')
    dataset_res_adult = pd.read_csv(file_path_adult+'/'+filename)

    dataset_housing = pd.read_csv(path+'data/housing/housing.csv')
    dataset_res_housing = pd.read_csv(file_path_housing+'/'+filename)

    if count_records:
        denom_adult = len(dataset_adult.values)
        denom_housing = len(dataset_housing.values)
        y_label = 'Suppressed records'
    else:
        denom_adult = len(dataset_adult.columns)
        denom_housing = len(dataset_housing.columns)
        y_label = 'Suppressed attributes'

    plot_path = path+"data/result/plots/1_1_a_"+y_label+".jpg"

    frequencies = []
    epsilons = []
    datasets = []

    for eps in dataset_res_adult.columns:
        freqs_adult = dataset_res_adult[eps]/denom_adult
        frequencies += list(freqs_adult)
        epsilons += [float(eps)]*len(freqs_adult)
        datasets += ['Adult']*len(freqs_adult)

        freqs_housing = dataset_res_housing[eps] / denom_housing
        frequencies += list(freqs_housing)
        epsilons += [float(eps)] * len(freqs_housing)
        datasets += ['Housing'] * len(freqs_housing)


    array = np.array([epsilons, frequencies]).T

    df = pd.DataFrame(array, columns=['ε', y_label])
    df['Dataset'] = datasets

    ax = sns.lineplot(x='ε', y=y_label, hue='Dataset', data=df,
                      palette=sns.xkcd_palette(['teal', 'orange', 'deep pink']))
    ax.set(ylim=(0, 1.05))
    #plt.show()
    plt.savefig(plot_path)
    plt.clf()

    return


def prepare_and_plot_1_1_b(dataset_name, epsilon):
    native_countries = []
    frequencies = []

    num_countries = 41

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    if epsilon == 'original':
        plot_path = path+'data/result/plots/needs_editing/' \
                    '1_1_b_original_'+dataset_name+'_native_country.jpg'

        file = path+'data/'+dataset_name+'/'+dataset_name+'.csv'
        dataset = pd.read_csv(file)
        native_freqs = dataset.groupby('native_country').groups
        for country in native_freqs:
            native_countries.append(country)
            frequencies.append(len(native_freqs[country]))
    else:
        plot_path = path+'data/result/plots/needs_editing/' \
                    '1_1_b_eps-'+str(epsilon)+'_'+ dataset_name + '_native_country.jpg'

        filepath = path+'data/result/sc_test/1_1/' + dataset_name + \
                   '/datasets'

        files = [file for file in os.listdir(filepath) if re.match("^eps-"+str(epsilon), file)]

        for file in files:
            dataset = pd.read_csv(filepath+'/'+file)
            native_freqs = dataset.groupby('native_country').groups
            for country in native_freqs:
                native_countries.append(country)
                frequencies.append(len(native_freqs[country]))

    array = np.array([frequencies]).T
    df = pd.DataFrame(array, columns=['Frequency'])
    df['Native country'] = native_countries

    united_states_index = native_countries.index('United-States')
    colors = ['cornflower blue']*num_countries
    colors[united_states_index] = 'tomato'

    ax = sns.barplot(x='Native country', y='Frequency', data=df, palette=sns.xkcd_palette(colors))
    ax.set(yscale='log')
    ax.set(ylim=(1, 30000))
    #plt.show()
    plt.savefig(plot_path)
    plt.clf()

    return


def prepare_and_plot_1_2(dataset_name):
    deltas = ['1.0E-5', '1.0E-6', '1.0E-7', '1.0E-8', '1.0E-9', '1.0E-10', '1.0E-11', '1.0E-12', '1.0E-13',
              '1.0E-14', '1.0E-15', '1.0E-16', '1.0E-17', '1.0E-18', '1.0E-19', '1.0E-20']
    deltas.reverse()
    if dataset_name == 'adult' or dataset_name == 'housing':
        ks = list(range(200, 4701, 300))
    else:
        raise RuntimeError("Does not recognise dataset", dataset_name)

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    file_path_safepub = path+'data/result/safepub_test/1_2/'+dataset_name
    file_path_sc = path+'data/result/sc_test/1_2/'+dataset_name
    pattern = "^norm_result_"

    plot_path = path+'data/result/plots/needs_editing/' \
                    '1_2_'+dataset_name+'.jpg'

    files_safepub = [file for file in os.listdir(file_path_safepub) if re.match(pattern, file)]
    files_sc = [file for file in os.listdir(file_path_sc) if re.match(pattern, file)]

    information_loss = []
    models = []
    parameters = []

    for file in files_sc:
        k = int(re.split(pattern+"k_|.csv", file)[1])
        res_data = pd.read_csv(file_path_sc+'/'+file)
        data = res_data['sse']
        information_loss += list(data)
        models += ['MicroDP']*len(data)
        parameters += [k]*len(data)

    for file in files_safepub:
        delta = re.split(pattern+"delta_|.csv", file)[1]
        res_data = pd.read_csv(file_path_safepub+'/'+file)
        data = res_data['sse']
        information_loss += list(data)
        models += ['SafePub']*len(data)
        parameters += [ks[deltas.index(delta)]]*len(data)

    array = np.array([information_loss, parameters]).T

    df = pd.DataFrame(data=array, columns=['Information loss', 'k/𝛿'])
    df['Model'] = models

    ax = sns.lineplot(x='k/𝛿', y='Information loss', hue='Model', style='Model', data=df,
                      palette=sns.xkcd_palette(['windows blue', 'amber']), markers=['o', 'o'], dashes=False)
    ax.set(ylim=(0.0, 0.75))
    #plt.show()
    plt.savefig(plot_path)
    plt.clf()


def prepare_and_plot_2_1(dataset_name):
    match = "^norm_result_"

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    safepub_path = path+"data/result/safepub_test/2_1/" + dataset_name

    sc_path = path+"data/result/sc_test/2_1/" + dataset_name

    plot_path = path+"data/result/plots/2_1_" + dataset_name + ".jpg"

    if dataset_name == 'adult':
        attribute_range = list(range(2, 9))
    elif dataset_name == 'housing':
        attribute_range = list(range(2, 10))
    elif dataset_name == 'musk':
        attribute_range = list(range(2, 21))
    else:
        raise RuntimeError("Does not recognise dataset", dataset_name)

    num_attributes = []
    information_loss = []
    models = []


    for a in attribute_range:
        sc_files = [file for file in os.listdir(sc_path+'/'+str(a)) if re.match(match, file)]
        safepub_files = [file for file in os.listdir(safepub_path+'/'+str(a)) if re.match(match, file)]

        for file in sc_files:
            df = pd.read_csv(sc_path+'/'+str(a)+'/'+file)
            data = list(df['sse'])
            information_loss += data
            num_attributes += [a]*len(data)
            models += ['MicroDP'] * len(data)

        for file in safepub_files:
            df = pd.read_csv(safepub_path+'/'+str(a) + '/' + file)
            data = list(df['sse'])
            information_loss += data
            num_attributes += [a] * len(data)
            models += ['SafePub'] * len(data)

    array = np.array([num_attributes, information_loss]).T

    df = pd.DataFrame(array, columns=['Number of attributes', 'Information loss'])
    df['Model'] = models

    ax = sns.lineplot(x='Number of attributes', y='Information loss', hue='Model', data=df,
                      palette=sns.xkcd_palette(['windows blue', 'amber']))
    #ax.set(ylim=(0.0, 1.0))
    ax.set(yscale='log')
    plt.show()
    #plt.savefig(plot_path)
    plt.clf()

    return


def prepare_and_plot_2_1_all_datasets(model_name):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    result_path = re.search(pattern, path_in).group(0) + 'data/result/'+model_name+'_test/2_1'

    datasets = ['adult', 'housing', 'musk']
    names = {'adult': 'Adult', 'housing':'Housing', 'musk': 'Musk'}
    attribute_range = {'adult': range(2,9), 'housing': range(2,10), 'musk': range(2,21)}

    information_loss = []
    dataset_name = []
    num_attributes = []

    pattern = "^norm_result_"

    for dataset in datasets:
        res_path = result_path+'/'+dataset
        for a in attribute_range[dataset]:
            result_files = [file for file in os.listdir(res_path + '/' + str(a)) if re.match(pattern, file)]

            for file in result_files:
                df = pd.read_csv(res_path + '/' + str(a) + '/' + file)
                data = list(df['sse'])
                information_loss += data
                num_attributes += [a] * len(data)
                dataset_name += [names[dataset]] * len(data)

    array = np.array([num_attributes, information_loss]).T

    df = pd.DataFrame(array, columns=['Number of attributes', 'Information loss'])
    df['Dataset'] = dataset_name

    ax = sns.lineplot(x='Number of attributes', y='Information loss', hue='Dataset', data=df)

    ax.set(ylim=(0.2, 1))
    plt.show()
    #plt.savefig(plot_path)
    plt.clf()

    return

def prepare_and_plot_3_1(dataset_name):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    result_path = re.search(pattern, path_in).group(0)+'data/result/'

    epsilons = ['1.0', '2.0']

    record_linkage = []
    sse = []
    model = []
    epsis = []

    plot_path = result_path+'/plots/needs_editing/3_1_'+dataset_name+'.jpg'

    k_result_sse = pd.read_csv(result_path + 'k-anonym_test/3_1/' + dataset_name + '/norm_result_k5.csv')['sse']
    k_result_rl = pd.read_csv(result_path + 'k-anonym_test/3_1/' + dataset_name + '/result_k5.csv')['record_linkage']

    # k_record_linkage = list(k_result_rl.values*1000)
    # k_sse = list(k_result_sse.values)
    # k_model = ['k-anonymisation'] * len(k_result_rl.values)

    record_linkage += list((k_result_rl.values * 1000).astype(int))
    sse += list(k_result_sse.values)
    model += ['k-anonymisation'] * len(k_result_rl.values)
    epsis.append(3)

    for eps in epsilons:
        sc_result_sse = pd.read_csv(result_path+'sc_test/3_1/'+dataset_name+'/norm_result_eps-'+eps+'.csv')['sse']
        sc_result_rl = pd.read_csv(result_path+'sc_test/3_1/'+dataset_name+'/result_eps-'+eps+'.csv')['record_linkage']

        safepub_result_sse = pd.read_csv(result_path + 'safepub_test/3_1/' + dataset_name +
                                          '/norm_result_granularity_eps-' + eps + '.csv')['sse']
        safepub_result_rl = pd.read_csv(result_path + 'safepub_test/3_1/' + dataset_name +
                                        '/result_granularity_eps-' + eps + '.csv')['record_linkage']

        record_linkage += list((sc_result_rl.values*1000).astype(int))
        sse += list(sc_result_sse.values)
        model += ['MicroDP'] * len(sc_result_rl.values)
        epsis += [float(eps)]*len(sc_result_rl.values)
#
        record_linkage += list((safepub_result_rl.values*1000).astype(int))
        sse += list(safepub_result_sse.values)
        model += ['SafePub'] * len(safepub_result_rl.values)
        epsis += [float(eps)] * len(safepub_result_rl.values)

    array = np.array([record_linkage, sse, epsis]).T

    df = pd.DataFrame(np.array([record_linkage, sse, epsis]).T, columns=['Record linkage', 'Information loss', 'ε'])
    df['Model'] = model


    #k_df = pd.DataFrame(np.array([k_record_linkage, k_sse]).T, columns=['Record linkage', 'Information loss'])
    #k_df['Model'] = k_model

    ax = sns.barplot(x='ε', y='Record linkage', hue='Model', data=df,
                      palette=sns.xkcd_palette(['faded green', 'windows blue', 'amber']))
    #k_ax = sns.scatterplot(x='Record linkage', y='Information loss', hue='Model', data=k_df,
    #                       palette=sns.xkcd_palette(['faded green']))

    #ax.set(ylim=(0.0, 1))
    #k_ax.set(xlim=(0.0, 10))


    #plt.show()
    plt.savefig(plot_path)
    plt.clf()

    return

def prepare_and_plot_3_1_info_loss(dataset_name):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    result_path = re.search(pattern, path_in).group(0) + 'data/result/'

    epsilons = ['1.0', '2.0']

    record_linkage = []
    sse = []
    model = []
    epsis = []

    plot_path = result_path + '/plots/needs_editing/3_1_b_' + dataset_name + '.jpg'

    for eps in epsilons:
        sc_result_sse = pd.read_csv(result_path + 'sc_test/3_1/' + dataset_name + '/norm_result_eps-' + eps + '.csv')[
            'sse']
        sc_result_rl = pd.read_csv(result_path + 'sc_test/3_1/' + dataset_name + '/result_eps-' + eps + '.csv')[
            'record_linkage']

        safepub_result_sse = pd.read_csv(result_path + 'safepub_test/3_1/' + dataset_name +
                                         '/norm_result_granularity_eps-' + eps + '.csv')['sse']
        safepub_result_rl = pd.read_csv(result_path + 'safepub_test/3_1/' + dataset_name +
                                        '/result_granularity_eps-' + eps + '.csv')['record_linkage']

        record_linkage += list((sc_result_rl.values * 1000).astype(int))
        sse += list(sc_result_sse.values)
        model += ['MicroDP'] * len(sc_result_rl.values)
        epsis += [float(eps)] * len(sc_result_rl.values)
        #
        record_linkage += list((safepub_result_rl.values * 1000).astype(int))
        sse += list(safepub_result_sse.values)
        model += ['SafePub'] * len(safepub_result_rl.values)
        epsis += [float(eps)] * len(safepub_result_rl.values)

    array = np.array([record_linkage, sse, epsis]).T

    df = pd.DataFrame(np.array([record_linkage, sse, epsis]).T, columns=['Record linkage', 'Information loss', 'ε'])
    df['Model'] = model

    ax = sns.lineplot(x='Record linkage', y='Information loss', hue='Model', data=df,
                     palette=sns.xkcd_palette(['windows blue', 'amber']))
    # k_ax = sns.scatterplot(x='Record linkage', y='Information loss', hue='Model', data=k_df,
    #                       palette=sns.xkcd_palette(['faded green']))

    # ax.set(ylim=(0.0, 1))
    # k_ax.set(xlim=(0.0, 10))

    plt.show()
    # plt.savefig(plot_path)
    plt.clf()


def study_result_3_1(dataset_name, model, eps):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    result_path = re.search(pattern, path_in).group(0) + 'data/result/'

    if model == 'safepub':
        filename_sse = 'norm_result_granularity'
        filename_rl = 'result_granularity'
    else:
        filename_sse = 'norm_result'
        filename_rl = 'result'

    result_sse = pd.read_csv(result_path + model+'_test/3_1/' + dataset_name + '/'+filename_sse+'_eps-' + eps + '.csv')['sse']
    result_rl = pd.read_csv(result_path + model+'_test/3_1/' + dataset_name + '/'+filename_rl+'_eps-' + eps + '.csv')[
        'record_linkage']

    print("mean sse", np.mean(result_sse.values))
    print("std sse", np.std(result_sse.values))

    print("mean rl", np.mean(result_rl.values)*1000)
    print("std rl", np.std(result_rl.values)*1000)


if __name__ == '__main__':
    housing_attributes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                          "population", "households", "median_income", "median_house_value"]
    musk_attributes = [str(i+1) for i in range(20)]

    #test_1_1('housing', 'safepub')


    #test_2_1('musk', 'sc', musk_attributes) # Evaluate attributes 2, 3, 4

    #test_3_1('housing_small', 'sc')

    #prepare_and_plot_1_1('housing')

    #prepare_and_plot_1_1_a(True)
    #prepare_and_plot_1_1_a(False)

    #prepare_and_plot_1_1_b('adult', 'original')

    #prepare_and_plot_1_2('adult')

    #prepare_and_plot_2_1('adult')

    #prepare_and_plot_1_2('housing')
    #prepare_and_plot_1_2('adult')

    #prepare_and_plot_1_1_a()

    #prepare_and_plot_1_1('adult')

    #prepare_and_plot_1_1_safepub('housing')

    #prepare_and_plot_2_1_all_datasets('safepub')
    #prepare_and_plot_2_1('musk')
    #study_result_3_1('housing_small', 'sc', '2.0')

    pass