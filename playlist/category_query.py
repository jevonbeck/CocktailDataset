from typing import List, Dict, Any
import pandas as pd


# String Methods
def first_index_in_string(phrase: str, string: str) -> int:
    if not isinstance(string, str):
        return -1

    start_index = 0
    str_length = len(string)
    phrase_length = len(phrase)

    while start_index < str_length:
        try:
            start_index = string[start_index:].index(phrase) + start_index
        except ValueError:
            break

        before_char = string[start_index-1:start_index] if start_index != 0 else None
        if before_char is not None and before_char.isalpha():
            start_index += phrase_length
            continue

        end_index = start_index + phrase_length
        if end_index < str_length and string[end_index:end_index+1].isalpha():
            start_index += phrase_length
            continue

        return start_index
    return -1


def is_phrase_in_string(phrase: str, string: str) -> bool:
    return first_index_in_string(phrase, string) >= 0


def ingredient_str_has_at_least_one_phrase(ingredient: str, contain_phrases: List[str]) -> bool:
    for phrase in contain_phrases:
        if is_phrase_in_string(phrase, ingredient):
            return True

    return False


def ingredient_str_has_at_least_one_phrase_from_each_list(ingredient: str, contain_lists: List[List[str]]):
    for contain_list in contain_lists:
        if not ingredient_str_has_at_least_one_phrase(ingredient, contain_list):
            return False

    return True


# Series Methods
def ingredient_has_at_least_one_phrase(ingredient: pd.Series,
                                       contain_phrases: List[str]) -> pd.Series:
    # Cheat for now
    return ingredient.apply(ingredient_str_has_at_least_one_phrase, args=(contain_phrases,))


def ingredient_has_at_least_one_phrase_from_each_list(ingredient: pd.Series,
                                                      contain_lists: List[List[str]]) -> pd.Series:
    has_phrase = pd.Series(True, index=ingredient.index)
    for contain_list in contain_lists:
        has_phrase = has_phrase & ingredient_has_at_least_one_phrase(ingredient, contain_list)

    return has_phrase


class CategoryQuery:
    def __init__(self, name: str, contains: List[str] = None,
                 not_contains: List[str] = None,
                 contains_and: List[List[str]] = None,
                 attributes: Dict[str, Any] = None):
        self.name: str = name
        self.contains: List[str] = contains
        self.not_contains: List[str] = not_contains
        self.contains_and: List[List[str]] = contains_and
        self.attributes: Dict[str, Any] = attributes

    def __getattr__(self, item):
        if self.attributes:
            return self.attributes.get(item)
        return None

    # String Methods
    def ingredient_str_is_in_category(self, multi_ingredients: str) -> bool:
        ingredients = multi_ingredients.split(',')
        for ingredient in ingredients:
            is_in_category = False

            has_contains = self.contains is not None
            has_contains_and = self.contains_and is not None

            if not has_contains and not has_contains_and:
                has_contains = True
                contain_phrases = [self.name]
            elif has_contains:
                contain_phrases = self.contains.copy()
                contain_phrases.append(self.name)
            else:
                contain_phrases = None

            if has_contains:
                is_in_category = ingredient_str_has_at_least_one_phrase(ingredient, contain_phrases)
            elif has_contains_and:
                is_in_category = ingredient_str_has_at_least_one_phrase_from_each_list(ingredient, self.contains_and)

            if self.not_contains is not None:
                is_in_category = is_in_category and not ingredient_str_has_at_least_one_phrase(
                    ingredient, self.not_contains
                )

            if is_in_category:
                return True
        return False

    # Series Methods
    def ingredient_is_in_category(self, ingredient: pd.Series) -> pd.Series:
        is_in_category = pd.Series(False, index=range(0, ingredient.shape[0]))

        has_contains = self.contains is not None
        has_contains_and = self.contains_and is not None
    
        if not has_contains and not has_contains_and:
            has_contains = True
            contain_phrases = [self.name]
        elif has_contains:
            contain_phrases = self.contains.copy()
            contain_phrases.append(self.name)
        else:
            contain_phrases = None

        is_multiple = ingredient.str.contains(',')
        is_not_multi = ~is_multiple

        # Handle single-ingredient strings
        if has_contains:
            is_in_category.loc[is_not_multi] = \
                ingredient_has_at_least_one_phrase(ingredient.loc[is_not_multi], contain_phrases)
        elif has_contains_and:
            is_in_category.loc[is_not_multi] = \
                ingredient_has_at_least_one_phrase_from_each_list(ingredient.loc[is_not_multi], self.contains_and)
        else:
            return is_in_category
    
        if self.not_contains is not None:
            is_in_category.loc[is_not_multi] = is_in_category.loc[is_not_multi] & \
                                               ~ingredient_has_at_least_one_phrase(
                                                   ingredient.loc[is_not_multi], self.not_contains
                                               )
        # Handle multi-ingredient strings
        if is_multiple.any():
            is_in_category.loc[is_multiple] = ingredient.loc[is_multiple].map(self.ingredient_str_is_in_category)
        return is_in_category
