# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    bin_op_0 = module_0.BinOp()
    base_node_transformer_0 = module_1.BaseNodeTransformer(bin_op_0)


def test_case_1():
    pass_0 = module_0.Pass()
    base_import_rewrite_0 = module_1.BaseImportRewrite(pass_0)
    list_0 = [base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [none_type_0, none_type_0, base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    str_0 = "%;"
    base_import_rewrite_0 = module_1.BaseImportRewrite(str_0)
    list_0 = [str_0, str_0, base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit(import_from_0)


def test_case_4():
    str_0 = ""
    base_import_rewrite_0 = module_1.BaseImportRewrite(str_0)
    list_0 = [str_0, str_0, base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
