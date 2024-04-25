from abc import ABCMeta, abstractmethod
from neura_modules.utils.data_type import DataType

class NeuraModule(metaclass=ABCMeta):
    def __init__(self):
        self._configs_reqd = []
        self._inputs_reqd = []
        self._outputs_available = []

        self.configs = {}


    @property
    def configs_reqd(self) -> list[tuple[str, DataType]]:
        return self._configs_reqd


    @configs_reqd.setter
    def configs_reqd(self, configs_reqd: list[tuple[str, DataType]]) -> None:
        self._configs_reqd = configs_reqd


    @property
    def inputs_reqd(self) -> list[str]:
        return self._inputs_reqd


    @inputs_reqd.setter
    def inputs_reqd(self, inputs_reqd: list[str]) -> None:
        self._inputs_reqd = inputs_reqd


    @property
    def outputs_available(self) -> list[str]:
        return self._outputs_available


    @outputs_available.setter
    def outputs_available(self, outputs_available: list[str]) -> None:
        self._outputs_available = outputs_available


    @abstractmethod
    def run_module(self, inputs: dict[str, object]) -> dict[str, any]:
        raise NotImplementedError()


    def setup_module(self, inputs: dict[str, object]) -> None:
        """
        Default implementation. Modules can override this function if needed.

        Arguments:
            inputs (dict[str: any])  - A dictionary containing all of the inputs for the module.
                                       The key is the variable name and the value is the variable's
                                       value.
        """
        pass


    def teardown_module(self) -> None:
        """
        Default implementation. Modules can override this function
        if needed.
        """
        pass


    def check_configs_valid(self, configs: list[str]) -> bool:
        configs_reqd_variable_names = [config[0] for config in self.configs_reqd]
        return self._check_all_required_items_exist(required_items=configs_reqd_variable_names,
                                                    given_items=configs)


    def check_inputs_valid(self, inputs: list[str]) -> bool:
        return self._check_all_required_items_exist(required_items=self.inputs_reqd,
                                                    given_items=inputs)


    def check_outputs_valid(self, outputs: list[str]) -> bool:
        return self._check_all_required_items_exist(required_items=outputs,
                                                    given_items=self.outputs_available)


    def set_configs(self, configs: dict):
        self.configs = configs


    def _check_all_required_items_exist(self, required_items: list[str], given_items: list[str]) -> bool:
        """
        This function checks that given_items contains all of the required_items.

        Arguments:
            required_items (list[str]) - List of required items.
            given_items (list[str])    - List of items that needs to have all of the
                                         required items.
        
        Returns:
            (bool) : True if given_items contains all of the required_items. False otherwise.
        """

        required_items = set(required_items)
        given_items = set(given_items)

        return required_items.issubset(given_items)