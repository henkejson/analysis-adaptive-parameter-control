# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    str_0 = module_0.Str()
    yield_from_transformer_0 = module_1.YieldFromTransformer(str_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    str_0 = module_0.Str()
    yield_from_transformer_0 = module_1.YieldFromTransformer(str_0)


def test_case_2():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_0 = [yield_from_transformer_0, none_type_0, none_type_0]
    except_handler_0 = module_0.ExceptHandler(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(except_handler_0)
