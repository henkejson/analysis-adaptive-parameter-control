# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    base_node_transformer_0 = module_0.BaseNodeTransformer(none_type_0)


def test_case_1():
    div_0 = module_1.Div()
    base_import_rewrite_0 = module_0.BaseImportRewrite(div_0)
    list_0 = [base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    div_0 = module_1.Div()
    base_import_rewrite_0 = module_0.BaseImportRewrite(div_0)
    str_0 = "!"
    list_0 = [div_0, str_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit(import_from_0)


def test_case_3():
    div_0 = module_1.Div()
    base_import_rewrite_0 = module_0.BaseImportRewrite(div_0)
    str_0 = ""
    list_0 = [div_0, str_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    div_0 = module_1.Div()
    base_import_rewrite_0 = module_0.BaseImportRewrite(div_0)
    base_import_rewrite_1 = module_0.BaseImportRewrite(div_0)
    list_0 = [base_import_rewrite_1, base_import_rewrite_1]
    str_0 = ""
    str_1 = "]Jna99w["
    ext_slice_0 = module_1.ExtSlice()
    base_node_transformer_0 = module_0.BaseNodeTransformer(ext_slice_0)
    dict_0 = {
        str_0: base_import_rewrite_1,
        str_0: base_import_rewrite_1,
        str_0: base_import_rewrite_1,
        str_1: div_0,
    }
    import_from_0 = module_1.ImportFrom(*list_0, **dict_0)
    none_type_0 = None
    list_1 = [none_type_0]
    import_from_1 = module_1.ImportFrom(*list_1)
    base_import_rewrite_1.visit_ImportFrom(import_from_1)
