# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    base_import_rewrite_0 = module_0.BaseImportRewrite(none_type_0)


def test_case_1():
    bool_op_0 = module_1.BoolOp()
    base_import_rewrite_0 = module_0.BaseImportRewrite(bool_op_0)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    bool_op_0 = module_1.BoolOp()
    base_import_rewrite_0 = module_0.BaseImportRewrite(bool_op_0)
    list_0 = [base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    bool_op_0 = module_1.BoolOp()
    base_import_rewrite_0 = module_2.walk(bool_op_0)
    list_0 = [base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_1 = module_0.BaseImportRewrite(bool_op_0)
    base_import_rewrite_1.visit_ImportFrom(import_from_0)


def test_case_4():
    bool_op_0 = module_1.BoolOp()
    var_0 = module_2.iter_fields(bool_op_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(bool_op_0)
    list_0 = [var_0, var_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
