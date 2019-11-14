import pytest

class PytestBase(): 

    def setup_method(self, test_method):
        print("/n" + "-"*80)
        print("%s:%s" % (type(self).__name__, test_method))

    def check_function_exists_in_object(self, object_to_test, function_name):
        function = getattr(object_to_test, function_name, None)
        assert function is not None
        assert callable(function)

    def check_attribute_or_property_exists_in_object(self, object_to_test, attribute_or_property_name): 
        attribute_or_property = getattr(object_to_test, attribute_or_property_name, None)
        assert attribute_or_property is not None
        