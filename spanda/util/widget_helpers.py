from ipywidgets import Checkbox, Layout, Button, Box, VBox, HBox
from IPython.display import display


## Checkbox widget
def checkbox_widget(input_list: list, value: bool = True):
    """Given a list of strings, create a checkbox widget for each string.

    Args:
        input_list (list):
            List of strings to create checkbox widgets for.
        value (bool, optional):
            Default value of each checkbox. Defaults to True.

    Returns:
        tuple:
            checkbox_widget (ipywidgets.VBox):
                Widget containing all checkboxes.
            return_items (function):
                A callback function that returns a list of the checkbox values.
    """

    def _create_checkbox_widget(input_list: list, value: bool = True):
        checkbox_layout = Layout(width="auto")
        checkboxes = [
            Checkbox(description=input, value=value, layout=checkbox_layout)
            for input in input_list
        ]
        return checkboxes

    def _on_checkbox_change(change):
        if change["name"] == "value" and change["new"] == True:
            print(f"{change['owner'].description} will be analyzed")
        elif change["name"] == "value" and change["new"] == False:
            print(f"{change['owner'].description} will NOT be analyzed")

    checkboxes = _create_checkbox_widget(input_list, value)
    for checkbox in checkboxes:
        checkbox.observe(_on_checkbox_change)

    def return_items():
        return [checkbox.value for checkbox in checkboxes]

    return VBox(checkboxes), return_items
