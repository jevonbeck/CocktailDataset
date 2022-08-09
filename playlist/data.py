from typing import List, Tuple
import pandas as pd

from playlist.category_query import CategoryQuery, first_index_in_string, ingredient_has_at_least_one_phrase

citrus_list = ['citrus', 'grapefruit', 'lemon', 'lime', 'mandarin', 'orange',  'yuzu']
nut_list = ['almond',  'hazelnut']
fruit_list = [
    'fruit',
    'acai berry', 'apple', 'apricot', 'banana', 'blackberry', 'cassis', 'cherry', 'coconut',
    'cranberry', 'grape', 'lychee', 'mango', 'maraschino', 'melon', 'passion fruit', 'peach',
    'pear', 'peche', 'pineapple', 'pomegranate', 'raspberry', 'strawberry',
    'tropical fruit'
] + nut_list
floral_list = ['floral', 'rose']
herb_spice_list = ['herbal', 'bison grass', 'cinnamon', 'ginger', 'jalapeno', 'menthe', 'mint',
                   'peppermint', 'vanilla'] + floral_list
milky_list = ['cream', 'milk']

dessert_list = ['dessert', 'butterscotch', 'chocolate', 'coffee', 'gingerbread']
syrup_list = ['agave', 'extract', 'gomme', 'grenadine', 'honey', 'orgeat']

liqueur_list = ['liqueur', 'schnapps', 'creme de']
flavoured_spirits = ['spirit', 'bourbon', 'brandy', 'gin', 'rum', 'tequila', 'vodka', 'wine'] + liqueur_list
juice_list = ['juice',  'nectar', 'puree']

brand_type_map = {
    'alco-pop': ['gin & ginger ale'],
    'almond liqueur': ['amaretto', 'creme de noyaux'],
    'apple brandy': ['applejack', 'calvados'],
    'cherry brandy': ['kirschwasser'],
    'cherry liqueur': ['cherry heering'],
    'citrus juice': ['collins mix', 'margarita mix', 'lemonade'],
    'cream liqueur': ['advocaat', 'crema di limoncello', 'eggnog', 'irish cream'],  # exception in alphabetic ordering
    # to substitute 'crema di limoncello' before 'limoncello'
    'citrus liqueur': ['limoncello', 'mandarine napoleon liqueur'],
    'dessert liqueur': ['creme de cacao'],
    'dry vermouth': ['lillet blanc', 'punt e mes'],
    'floral liqueur': ['creme de violette', 'creme yvette', 'elderflower liqueur', 'st germain'],
    'fruit juice': ['v8 cocktail juice'],
    'fruit spirit': ["pimm's", 'rock and rye'],
    'herbal spirit': ['absinthe', 'allspice liqueur', 'amaro', 'anis', 'anisette', 'aperol', 'b & b', 'becherovka',
                      'benedictine', 'campari', 'chartreuse', 'cynar', 'drambuie', 'dubonnet', 'falernum',
                      'fernet-branca', 'galliano', 'jagermeister', 'jägermeister', 'kummel', 'pastis',
                      'strega', 'sambuca', 'swedish punch', 'swedish punsch', 'jalepeno-infused tequila'],
    'orange brandy': ['grand marnier'],
    'orange liqueur': ['amer', 'blue curacao', 'cointreau', 'creole shrubb', 'curacao', 'triple sec'],
    'over-proof rum': ['wray and nephew'],
    'red wine': ['claret'],
    'soda': ['7-up', 'carbonated water', 'cola', 'ginger ale', 'ginger beer', 'mountain dew',
             'root beer', 'tonic', 'tonic water'],
    'sparkling wine': ["moscato d'asti"],
    'sweet sherry': ['palo cortado', 'pedro ximenez'],
    'sweet vermouth': ['lillet rouge'],
}

drink_category_list = [
    CategoryQuery('alco-pop', attributes={'alco_factor': 0.125}),
    CategoryQuery('beer', contains=['ale', 'cider', 'stout'], attributes={'alco_factor': 0.125}),
    CategoryQuery('bitters', contains=['tiki'], attributes={'alco_factor': 1}),
    CategoryQuery('brandy', contains=['armagnac', 'cognac', 'pisco'],
                  not_contains=(fruit_list + citrus_list + dessert_list), attributes={'alco_factor': 1}),
    CategoryQuery('citrus juice', contains_and=[citrus_list, juice_list]),
    CategoryQuery('citrus spirit', contains_and=[citrus_list, flavoured_spirits], attributes={'alco_factor': 0.5}),
    CategoryQuery('cream spirit', contains_and=[milky_list, flavoured_spirits], attributes={'alco_factor': 0.5}),
    CategoryQuery('dessert spirit', contains_and=[dessert_list, flavoured_spirits], attributes={'alco_factor': 0.5}),
    CategoryQuery('egg', contains=['aquafaba']),
    CategoryQuery('flavoured water', contains_and=[['water'], herb_spice_list + citrus_list]),
    CategoryQuery('floral spirit', contains_and=[floral_list, flavoured_spirits], attributes={'alco_factor': 0.5}),
    CategoryQuery('fortified wine', contains=['madeira', 'port', 'sherry', 'vermouth'],
                  attributes={'alco_factor': 1}),
    CategoryQuery('fruit juice', contains=juice_list,
                  not_contains=(citrus_list + syrup_list + ['tomato'])),
    CategoryQuery('fruit spirit', contains_and=[fruit_list, flavoured_spirits], not_contains=['soaked'],
                  attributes={'alco_factor': 0.5}),
    CategoryQuery('gin', not_contains=(fruit_list + citrus_list + dessert_list + herb_spice_list),
                  attributes={'alco_factor': 1}),
    CategoryQuery('herbal spirit', contains_and=[herb_spice_list + syrup_list + ['herbal'], flavoured_spirits],
                  attributes={'alco_factor': 0.5}),
    CategoryQuery('milk', contains=['cream', 'half-and-half', 'yogurt'],
                  not_contains=(['irish', 'sherry', 'whipped'] + flavoured_spirits)),
    CategoryQuery('rum', contains=['cachaca', 'rhum'], not_contains=(fruit_list + citrus_list),
                  attributes={'alco_factor': 1}),
    CategoryQuery('sauce', contains=['brine', 'catsup', 'ketchup', 'tomato', 'vinegar']),
    CategoryQuery('soda', not_contains=flavoured_spirits),
    CategoryQuery('sparkling wine', contains=['cava', 'champagne', 'prosecco'], attributes={'alco_factor': 0.25}),
    CategoryQuery('syrup', contains=syrup_list, not_contains=flavoured_spirits),
    CategoryQuery('tea', contains=['coffee'], not_contains=flavoured_spirits),
    CategoryQuery('tequila', contains=['mezcal'], not_contains=herb_spice_list, attributes={'alco_factor': 1}),
    CategoryQuery('water', not_contains=(['soda', 'tonic'] + herb_spice_list + citrus_list)),
    CategoryQuery('vodka', contains=['aquavit'], not_contains=(fruit_list + citrus_list + herb_spice_list),
                  attributes={'alco_factor': 1}),
    CategoryQuery('whisky', contains=['bourbon', 'whiskey', 'scotch'], not_contains=(['bitters'] + herb_spice_list),
                  attributes={'alco_factor': 1}),
    CategoryQuery('wine', contains=['sake'], not_contains=(['fortified', 'port', 'sparkling'] + fruit_list),
                  attributes={'alco_factor': 0.25}),
]

strain_list = ['strain']
stir_list = ['stir']
build_list = ['top with', 'fill'] + stir_list

method_category_list = [
    CategoryQuery('blend', contains=['crushed ice', 'blender']),
    CategoryQuery('shake', contains=['roll'], not_contains=["don't shake"]),
    CategoryQuery('stir-strain', contains_and=[stir_list + ['martini'], strain_list + ['martini']],
                  not_contains=["don't stir", 'shake']),
    CategoryQuery('build', contains=build_list, not_contains=(strain_list + ["don't stir"])),
]

no_measure_col = 'no-measurement'
glass_total_col = 'glass-total'
total_col = 'total-liquid'
ingr_count_col = 'total-ingredients'
GLASS_FILL_PERCENT = 0.75
oz_in_ml = 30


def get_all_ingredients(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    result = pd.Series([], dtype='str')
    for col in cols:
        result = result.append(df[col].str.strip().str.lower())

    return pd.Series(result[result.notna()].unique()).sort_values()


def clean_up_data(df: pd.DataFrame, cols: List[Tuple[str, str]]):
    for ingredient_col, measurement_col in cols:
        ingredient = df[ingredient_col]
        measurement = df[measurement_col]

        # Take ingredient wrongly placed in measurement and place in ingredient
        is_empty_ingredient = ingredient.isna() & measurement.notna()
        ingredient.loc[is_empty_ingredient] = measurement.loc[is_empty_ingredient]
        measurement.loc[is_empty_ingredient] = None  # set to top-up since ingredient already obtained

        # Take measurement wrongly placed in ingredient and place in measurement
        ingred_has_measurement = ingredient_has_at_least_one_phrase(ingredient, [
            'oz', 'dash', 'dashes', 'splash', 'splashes', 'bottle', 'bottles', '750-ml',
            'tsp', 'tsps', 'c', 'eggs', 'egg'
        ])
        is_ingred_measurement = measurement.isna() & ingred_has_measurement
        measurement.loc[is_ingred_measurement] = ingredient.loc[is_ingred_measurement]\
            .str.replace('eggs', '').str.replace('egg', '')

        # Cleanup
        ingredient.fillna('', inplace=True)
        measurement.fillna('', inplace=True)
        ingredient = ingredient.str.strip().str.lower()
        measurement = measurement.str.strip().str.lower()

        # Data substitution
        needs_maraschino_liqueur_sub = ingredient == 'maraschino'
        ingredient.loc[needs_maraschino_liqueur_sub] = 'maraschino liqueur'

        is_bitters = ingredient.str.contains('bitters') | ingredient.str.contains('tiki')
        has_ozs = measurement.str.contains('oz')
        sub_bitters_measure = is_bitters & has_ozs
        measurement.loc[sub_bitters_measure] = measurement.loc[sub_bitters_measure].str.replace('oz', 'dash')

        # add updated cols to df
        fixed_ingredient_col = 'fixed-' + ingredient_col
        fixed_measurement_col = 'fixed-' + measurement_col
        df[fixed_ingredient_col] = ingredient
        df[fixed_measurement_col] = measurement


def replace_phrase_in_string(string: str, phrase: str, replacement: str) -> str:
    start_index = first_index_in_string(phrase, string)
    return string[0:start_index] + replacement + string[start_index + len(phrase):] if start_index >= 0 else string


def substitute_brands(ingredient: pd.Series):
    result = ingredient.copy()
    for replacement, phrases in brand_type_map.items():
        for phrase in phrases:
            has_phrase = result.str.contains(phrase)
            # Cheat for now
            result.loc[has_phrase] = result.loc[has_phrase].apply(replace_phrase_in_string,
                                                                  args=(phrase, replacement))
    return result


def _handle_or_in_multiple(result: pd.Series):
    has_or = result.str.contains(' or ')
    if has_or.any():
        result.loc[has_or] = result.loc[has_or].str.replace(' or ', ',')
        result.loc[has_or] = result.loc[has_or].apply(lambda x: x.split(',')[0])


def standardise_multiple(ingredients: pd.Series):
    result = ingredients.copy()

    has_multiple = result.str.contains(',') | result.str.contains(';') | result.str.contains(' and ')
    result.loc[has_multiple] = result.loc[has_multiple].str.replace(' and ', ',').str.replace(';', ',')

    _handle_or_in_multiple(result)

    return result


def _get_unit(col_name: str):
    if col_name == 'egg':
        return 1.5 * oz_in_ml  # 1 medium egg == 1.5 oz
    if col_name == 'citrus juice':
        return oz_in_ml  # 1 lime == 2 tbsp, 1 lemon == 2-3 tbsp (1 tbsp approx 17 ml)

    return 1


def _parse_measurement(measure_value: str, unit: str):
    if unit != '' and unit in measure_value:
        if measure_value == unit:
            return 1
        try:
            return float(measure_value[0:measure_value.index(unit)])
        except ValueError:
            return 0
    return 0


def standardise_measurement_str(measurement: str, col_name: str):
    if measurement == '':
        return 0

    measure_value = measurement.replace(' 1/2', '.5').replace('1/2', '0.5')\
        .replace(' 1/4', '.25').replace('1/4', '0.25')\
        .replace(' 3/4', '.75').replace('3/4', '0.75') \
        .replace(' 1/3', '.333').replace('1/3', '0.333') \
        .replace(' 2/3', '.667').replace('2/3', '0.667') \
        .replace('splash', 'dash').replace('750-ml', 'bottle') \
        .replace('ounce', 'oz').replace('ozs', 'oz')
    result = measure_value

    if ' or ' in result:
        result = result.replace(' or ', ',')
        result = result.split(',')[0]

    unit = ''
    multiplier = _get_unit(col_name)

    if 'oz' in result:
        multiplier = oz_in_ml  # ml
        result = result.replace('oz', '')
        unit = 'oz'
    elif 'bottle' in result:
        multiplier = 750  # ml
        result = result.replace('bottles', '').replace('bottle', '')
        unit = 'bottle'
    elif 'dash' in result:
        multiplier = 2  # ml
        result = result.replace('dashes', '').replace('dash', '')
        unit = 'dash'
    elif 'tsp' in result:
        multiplier = 5  # ml
        result = result.replace('tsps', '').replace('tsp', '')
        unit = 'tsp'
    elif 'c' in result:  # 'c' is for cup
        multiplier = 240  # ml
        result = result.replace('c', '')
        unit = 'c'

    try:
        result = float(result)
        multiplier = 0 if unit == '' and multiplier == 1 else multiplier
    except ValueError:
        result = _parse_measurement(measure_value, unit)

    return round(result * multiplier, 1)


def _get_glass_size(glass_size_str: str):
    temp = glass_size_str.strip().split(' ')
    size_approx = str((float(temp[0]) + float(temp[-2]))/2) if len(temp) > 2 and temp[-2].isnumeric() else temp[0]
    new_glass_size_str = size_approx + ' ' + temp[-1]
    return standardise_measurement_str(new_glass_size_str, 'not-important')


def standardise_measurement(measurement: pd.Series, col_name: str) -> pd.Series:
    measure_value = measurement.str.replace(' 1/2', '.5').str.replace('1/2', '0.5') \
        .str.replace(' 1/4', '.25').str.replace('1/4', '0.25') \
        .str.replace(' 3/4', '.75').str.replace('3/4', '0.75') \
        .str.replace(' 1/3', '.333').str.replace('1/3', '0.333') \
        .str.replace(' 2/3', '.667').str.replace('2/3', '0.667') \
        .str.replace('splash', 'dash').str.replace('750-ml', 'bottle') \
        .str.replace('ounce', 'oz').str.replace('ozs', 'oz')
    result = measure_value.copy()

    _handle_or_in_multiple(result)

    unit = pd.Series('', index=measure_value.index)
    multiplier = pd.Series(_get_unit(col_name), index=measure_value.index)

    has_oz = measure_value.str.contains('oz')
    multiplier.loc[has_oz] = oz_in_ml  # ml
    result.loc[has_oz] = measure_value.loc[has_oz].str.replace('oz', '')
    unit.loc[has_oz] = 'oz'

    has_bottle = measure_value.str.contains('bottle')
    multiplier.loc[has_bottle] = 750  # ml
    result.loc[has_bottle] = measure_value.loc[has_bottle].str.replace('bottles', '').str.replace('bottle', '')
    unit.loc[has_bottle] = 'bottle'

    has_dash = measure_value.str.contains('dash')
    multiplier.loc[has_dash] = 2  # ml
    result.loc[has_dash] = measure_value.loc[has_dash].str.replace('dashes', '').str.replace('dash', '')
    unit.loc[has_dash] = 'dash'

    has_tsp = measure_value.str.contains('tsp')
    multiplier.loc[has_tsp] = 5  # ml
    result.loc[has_tsp] = measure_value.loc[has_tsp].str.replace('tsps', '').str.replace('tsp', '')
    unit.loc[has_tsp] = 'tsp'

    has_cup = measure_value.str.contains('c')
    multiplier.loc[has_cup] = 240  # ml
    result.loc[has_cup] = measure_value.loc[has_cup].str.replace('c', '')
    unit.loc[has_cup] = 'c'

    result = pd.to_numeric(result, errors='coerce')

    invalid_results = result.isna()
    if invalid_results.any():
        temp = measure_value.loc[invalid_results].copy()
        temp.name = 'measure'
        temp = temp.to_frame()
        temp['unit'] = unit.loc[invalid_results]

        result.loc[invalid_results] = temp.apply(
            lambda x: _parse_measurement(x['measure'], x['unit']),
            axis=1
        )
    else:
        has_no_unit = (unit == '') & (multiplier == 1)
        multiplier.loc[has_no_unit] = 0

    return (result * multiplier).round(1)


def get_multi_measurement(ingredient_series: pd.Series, query: CategoryQuery,
                          ingredient_col: str, measurement_col: str):
    if ingredient_series.shape[0] == 0:
        return 0

    measurement = ingredient_series[measurement_col]
    ingredients = ingredient_series[ingredient_col].split(',')

    result = 0
    for ingredient in ingredients:
        if query.ingredient_str_is_in_category(ingredient):
            result += standardise_measurement_str(measurement, query.name)

    return result


def populate_drink_vectors(df: pd.DataFrame, cols: List[Tuple[str, str]]):
    df[no_measure_col] = ''
    df[total_col] = 0
    for query in drink_category_list:
        category_name = query.name
        df[category_name] = 0
        for ingredient_col, measurement_col in cols:
            ingredient = df[ingredient_col]
            is_multiple = ingredient.str.contains(',')
            is_in_cat = query.ingredient_is_in_category(ingredient)
            is_direct_swap = is_in_cat & ~is_multiple
            is_multi_swap = is_in_cat & is_multiple

            df.loc[is_direct_swap, category_name] += standardise_measurement(
                df.loc[is_direct_swap, measurement_col], category_name
            )
            df.loc[is_multi_swap, category_name] += df.loc[is_multi_swap, [ingredient_col, measurement_col]]\
                .apply(get_multi_measurement, args=(query, ingredient_col, measurement_col), axis=1)
            df.loc[is_in_cat & (df[category_name] == 0), no_measure_col] += ',' + category_name

        df[total_col] += df[category_name]

    has_no_measure = df[no_measure_col] != ''
    df.loc[has_no_measure, no_measure_col] = df.loc[has_no_measure, no_measure_col].str[1:]


def populate_method(df: pd.DataFrame, instructions_col: str):
    method = pd.Series('', index=df.index)
    for query in method_category_list:
        is_in_cat = query.ingredient_is_in_category(df[instructions_col])
        method.loc[is_in_cat] += ',' + query.name

    has_value = method.str.contains(',')
    method.loc[has_value] = method.loc[has_value].str[1:]
    method.loc[~has_value] = 'build'  # default value
    for query in method_category_list:
        df[query.name] = method.str.contains(query.name)


def _adjust_single_volumes(x: pd.Series):
    fill_vol = GLASS_FILL_PERCENT * x[glass_total_col]
    x.loc[x[no_measure_col]] += (fill_vol - x[total_col])
    x[total_col] = fill_vol
    return x


def _perform_glass_adjustments(df: pd.DataFrame):
    df[glass_total_col] = 0
    has_glass_size = df['glass-size'].notna()
    df.loc[has_glass_size, glass_total_col] = \
        df.loc[has_glass_size, 'glass-size'].apply(_get_glass_size)
    has_glass_total = df[glass_total_col] > 0
    is_within_glass_limits = has_glass_total & (df[total_col] <= GLASS_FILL_PERCENT * df[glass_total_col])

    is_single_no_measure = (df[no_measure_col] != '') & ~df[no_measure_col].str.contains(',')
    can_adjust_vol = is_within_glass_limits & is_single_no_measure
    df.loc[can_adjust_vol, :] = df.loc[can_adjust_vol, :].apply(_adjust_single_volumes, axis=1)

    excess_liquid = has_glass_total & (df[total_col] > 1.5 * GLASS_FILL_PERCENT * df[glass_total_col])
    for query in drink_category_list:
        category = query.name
        df.loc[excess_liquid, category] = \
            (df[category] * df[glass_total_col] / df[total_col]).loc[excess_liquid]
    df.loc[excess_liquid, total_col] = df.loc[excess_liquid, glass_total_col]


def _recompute_totals(df: pd.DataFrame):
    alcohol_col = 'total-alcohol'

    df[total_col] = 0
    df[alcohol_col] = 0
    df[ingr_count_col] = 0
    for query in drink_category_list:
        category_name = query.name

        has_ingredient = df[category_name] != 0
        df.loc[has_ingredient, ingr_count_col] = df.loc[has_ingredient, ingr_count_col] + 1

        df[total_col] += df[category_name]
        if query.alco_factor is not None and query.alco_factor > 0:
            df[alcohol_col] += df[category_name] * query.alco_factor

    df['alco-percent'] = df[alcohol_col] / df[total_col]


def _get_mr_boston_data():
    mr_boston_data = pd.read_csv('../data/mr-boston-flattened.csv')

    mr_boston_cols = [(f'ingredient-{x}', f'measurement-{x}') for x in range(1, 7)]
    clean_up_data(mr_boston_data, mr_boston_cols)

    instructions_col = 'fixed-instructions'
    mr_boston_data[instructions_col] = mr_boston_data['instructions'].str.strip().str.lower()
    populate_method(mr_boston_data, instructions_col)

    fixed_cols = [(f'fixed-ingredient-{x}', f'fixed-measurement-{x}') for x in range(1, 7)]
    for i_col, m_col in fixed_cols:
        mr_boston_data[i_col] = substitute_brands(mr_boston_data[i_col])
        mr_boston_data[i_col] = standardise_multiple(mr_boston_data[i_col])
    populate_drink_vectors(mr_boston_data, fixed_cols)
    _perform_glass_adjustments(mr_boston_data)
    _recompute_totals(mr_boston_data)

    ignore = (mr_boston_data[total_col] < oz_in_ml) | mr_boston_data[no_measure_col].str.contains(',') \
        | mr_boston_data[no_measure_col].str.contains('spirit')
    result = mr_boston_data[~ignore]
    return result


def get_data():
    mr_boston_data = _get_mr_boston_data()
    liquid_cols = [query.name for query in drink_category_list]
    return mr_boston_data, liquid_cols


def all_data_playground():
    mr_boston_data = _get_mr_boston_data()
    category_cols = [q.name for q in drink_category_list]

    ai_data = pd.read_csv('../data/all_drinks.csv')

    for cat_col in category_cols:
        has_multi = mr_boston_data[cat_col].str.contains(',')
        mr_boston_data[cat_col].loc[has_multi] = mr_boston_data[cat_col].loc[has_multi]  # todo: put apply here to convert to number

    mr_boston_i_cols = [f'fixed-ingredient-{x}' for x in range(1, 7)]
    ingredients = get_all_ingredients(mr_boston_data, mr_boston_i_cols)

    mr_boston_m_cols = [f'fixed-measurement-{x}' for x in range(1, 7)]
    measurements = get_all_ingredients(mr_boston_data, mr_boston_m_cols)

    original_ingredients = ingredients.copy()
    substitute_brands(ingredients)
    standardise_multiple(ingredients)
    ingredient_df = ingredients.to_frame(name='ingredients')
    ingredient_df['original_ingredient'] = original_ingredients
    ingredient_df['measure'] = (pd.Series(ingredient_df.index, index=ingredient_df.index) + 1)
    populate_drink_vectors(ingredient_df, [('ingredients', 'measure')])

    col_copy = ingredient_df.columns.to_list()
    col_copy.remove('original_ingredient')
    col_copy.remove('ingredients')
    col_copy.remove('measure')

    ingredient_df['alco_sum'] = 0
    for col in col_copy:
        ingredient_df['alco_sum'] += ingredient_df[col]

    non_liquid = ingredient_df[ingredient_df['alco_sum'] == 0]

    # Index(['name', 'category', 'measurement-1', 'ingredient-1', 'measurement-2',
    #        'ingredient-2', 'measurement-3', 'ingredient-3', 'measurement-4',
    #        'ingredient-4', 'measurement-5', 'ingredient-5', 'measurement-6',
    #        'ingredient-6', 'instructions', 'glass', 'glass-size'],
    #       dtype='object')
    return mr_boston_data

