# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast.ast3 as module_1
import typed_ast._ast3 as module_2


def test_case_0():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)


def test_case_2():
    str_0 = "tkintersimpledialog"
    yield_from_transformer_0 = module_0.YieldFromTransformer(str_0)
    var_0 = module_1.parse(str_0)
    var_1 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    none_type_0 = None
    module_0 = module_2.Module()
    yield_from_transformer_0 = module_0.YieldFromTransformer(module_0)
    str_0 = "\n    "
    dict_0 = {
        str_0: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
    }
    try_0 = module_2.Try(*dict_0)
    var_0 = yield_from_transformer_0.visit(try_0)
