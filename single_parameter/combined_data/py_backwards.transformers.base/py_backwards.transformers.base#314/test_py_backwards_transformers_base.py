# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    float_0 = 2534.8488
    base_node_transformer_0 = module_0.BaseNodeTransformer(float_0)


def test_case_1():
    str_0 = "J%,9n!HS@z"
    dict_0 = {str_0: str_0, str_0: str_0}
    base_import_rewrite_0 = module_0.BaseImportRewrite(str_0)
    keyword_0 = module_1.keyword(**dict_0)
    list_comp_0 = module_1.ListComp()
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_1.ImportFrom(*list_0, **dict_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    dict_0 = {}
    base_import_rewrite_0 = module_0.BaseImportRewrite(dict_0)
    list_0 = [base_import_rewrite_0, dict_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    str_0 = "Jd%,9n!HS@z"
    dict_0 = {str_0: str_0, str_0: str_0}
    keyword_0 = module_1.keyword(**dict_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(keyword_0)
    list_comp_0 = module_1.ListComp()
    base_import_rewrite_1 = module_0.BaseImportRewrite(list_comp_0)
    var_0 = module_2.iter_child_nodes(str_0)
    list_0 = [var_0, dict_0, list_comp_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_1.visit_ImportFrom(import_from_0)
