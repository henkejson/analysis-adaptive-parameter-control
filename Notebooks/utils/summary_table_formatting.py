import re
import pandas as pd


module_names_short = {
"codetiming._timer": 'timer',
"flutils.decorators": 'decorators',
"flutils.namedtupleutils": 'namedtupleutils',
'flutils.packages': 'packages',
'flutils.setuputils.cmd': 'cmd',
'httpie.output.formatters.headers': 'headers',
'httpie.plugins.base': 'h_base',
'mimesis.builtins.da': 'da',
'py_backwards.transformers.base': 'py_base',
'py_backwards.transformers.dict_unpacking': 'dict_unpacking',
'py_backwards.transformers.return_from_generator' : 'return_from_generator',
'py_backwards.transformers.yield_from': 'yield_from',
'py_backwards.utils.helpers': 'helpers',
'pymonet.immutable_list': 'immutable_list',
'pymonet.maybe': 'maybe', 
'pymonet.validation': 'validation',
'pypara.accounting.journaling': 'journaling',
'pytutils.lazy.lazy_import': 'lazy_import',
'pytutils.python': 'python',
'sanic.config': 'config',
'sanic.helpers': 'helpers',
'sanic.mixins.signals': 'signals',
'thonny.plugins.pgzero_frontend': 'pgzero_frontend',
'typesystem.tokenize.positional_validation': 'positional_validation'
}

parameter_names_short = {
'ChangeParameterProbability': 'ChangeParamProb',
'ChromosomeLength': 'ChromLen',
'CrossoverRate': 'Crossover',
'Elite': 'Elite',
'Population': 'Pop',
'RandomPerturbation': 'RandPert',
'StatementInsertionProbability': 'StatemInsertProb' ,
'TestChangeProbability': 'TestChangeProb',
'TestDeleteProbability': 'TestDeleteProb',
'TestInsertProbability': 'TestInsertProb',
'TestInsertionProbability': 'TestInsertionProb',
'TournamentSize': 'TourSize',
}




def extract_column_name(column):
        # Find all occurrences of text following 'T.'
        parts = re.findall(r'T\.([^\]]+)', column)
        return ' x '.join(parts)


def extract_column_names(columns):
    series = pd.Series([extract_column_name(col) for col in columns])
    return series

def replace_names(series, replacement_dict):
    series.replace({rf'\b{k}\b': v for k, v in replacement_dict.items()}, regex=True, inplace=True)


def get_replacement_list(columns):
    extracted_columns = extract_column_names(columns)
    replace_names(extracted_columns, module_names_short)
    replace_names(extracted_columns, parameter_names_short)
    return extracted_columns

def update_table(table, variable_name, replacement):
    pattern = re.compile(rf"{variable_name}\[(\d+)\]")

    def get_replacement_name(index):
        match = pattern.match(index)
        if match:
            index_num = int(match.group(1))  # Get the number inside the brackets
            if index_num in replacement.index:
                # Replace the index with the new descriptive name within the same pattern
                return f"{variable_name}[{replacement.loc[int(index_num)]}]"
        return index  # Return the original index if no match or no replacement

    # Update the DataFrame index
    table.index = [get_replacement_name(idx) for idx in table.index]





     

