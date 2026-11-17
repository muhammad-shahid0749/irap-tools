"""The browser's own form controls instead of Quasar's, for a compact, desktop-like look.

Each `create_*` function creates the control in the current NiceGUI container and returns its
element. The class `native-control` (`styles.css`) restores the browser's style, which
Tailwind's reset removes.

NiceGUI renders native tags in the render function of the page's root, so any update of the
page, e.g. a refreshed table, renders them again. Vue then sets a DOM property only if its prop
has changed since the last render, except `value`, which it sets on each render. A change by the
user is therefore stored in the element's props: for a select, so that a render does not restore
the initial `value`, and for a checkbox, so that `checked` matches the DOM and
`set_native_checkbox_checked` can set it back to an earlier state. A text input has no `value`
prop (see `create_native_input`), so that text that is being typed is not reset either.

When the user returns to a page, e.g. with Back, the server creates a new page, but the browser
restores the state of checkboxes, which the new page's state is not. Checkboxes and selects
therefore have `autocomplete="off"`, also in hand-written HTML (`NATIVE_CHOICE_ATTRIBUTES`). Text
inputs keep it on, so that the browser suggests earlier entries, which it does not restore
(checked in Edge).
"""

import json
import typing as T

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement

#: The attributes of a checkbox or select in hand-written HTML (see the module docstring).
NATIVE_CHOICE_ATTRIBUTES = 'class="native-control" autocomplete="off"'


def _on_user_change(element: ui.element, prop: str,
                    on_change: T.Callable[[T.Any], T.Any]) -> T.Callable[[T.Any], T.Any]:
    """Makes a handler of a change event that stores the new value in `element.props[prop]` (see
    the module docstring) before it calls `on_change`. The handler returns the result of
    `on_change`, so that NiceGUI awaits that of an async `on_change`."""
    def handle(event: T.Any) -> T.Any:
        element.props[prop] = event.args
        element.update()
        return on_change(event.args)

    return handle


def _create_field(label: str) -> ui.element:
    with ui.element("label").classes("field") as field:
        ui.label(label)
    return field


def _create_options(options: T.Mapping[str, str]) -> None:
    for option_value, text in options.items():
        TextElement(tag="option", text=text).props["value"] = option_value


def _create_select(label: str, value: str, on_change: T.Callable[[str], T.Any],
                   groups: T.Mapping[str | None, T.Mapping[str, str]]) -> ui.element:
    """
    Args:
        groups: Group label (`<optgroup>`), or None for options outside a group -> option value
            -> displayed text. The text of the selected option is the tooltip, which shows it
            when it is cut off, e.g. a long method name.
    """
    option_texts = {v: t for options in groups.values() for v, t in options.items()}
    with _create_field(label):
        select = ui.element("select").classes("native-control")
        # No tooltip for a value without an option, e.g. a reference method without scores,
        # for which the browser selects no option.
        select.props.update(value=value, title=option_texts.get(value, ""), autocomplete="off")
        with select:
            for group_label, options in groups.items():
                if group_label is None:
                    _create_options(options)
                    continue
                with ui.element("optgroup") as group:
                    _create_options(options)
                group.props["label"] = group_label

    def handle_change(new_value: str) -> T.Any:
        select.props["title"] = option_texts[new_value]  # Sent with the value.
        return on_change(new_value)

    select.on("change", _on_user_change(select, "value", handle_change),
              js_handler="(e) => emit(e.target.value)")
    return select


def create_native_select(label: str, options: T.Mapping[str, str], value: str,
                         on_change: T.Callable[[str], T.Any]) -> ui.element:
    """Creates a labeled `<select>`.

    Args:
        options: Value -> displayed text.
        on_change: Gets the new value. It can be async.
    """
    return _create_select(label, value, on_change, {None: options})


def create_native_grouped_select(label: str, groups: T.Mapping[str, T.Mapping[str, str]],
                                 value: str, on_change: T.Callable[[str], T.Any],
                                 ungrouped: T.Mapping[str, str] | None = None) -> ui.element:
    """Creates a labeled `<select>` with groups of options (`<optgroup>`).

    Args:
        groups: Group label -> option value -> displayed text.
        on_change: Gets the new value. It can be async.
        ungrouped: Options before the groups, e.g. one for none.
    """
    return _create_select(label, value, on_change, {None: ungrouped or {}, **groups})


def create_native_input(label: str, value: str, on_change: T.Callable[[str], None], *,
                        input_type: str = "text", placeholder: str = "",
                        size: int | None = None) -> ui.element:
    """Creates a labeled `<input>`. `on_change` gets the value when the input loses focus or
    Enter is pressed, which is before a click elsewhere is handled.

    The initial value is the input's `defaultValue`, not its `value`, so that a render of the page
    does not reset text that the user is typing (see the module docstring). The input cannot be set
    from code.

    Args:
        value: The initial value.
        on_change: Gets the new value. It can be async.
        input_type: The `type` attribute, e.g. 'number'.
        size: The width in characters, or None for the browser default.
    """
    element = _create_text_control("input", label, value, on_change, placeholder)
    element.props["type"] = input_type
    if size is not None:
        element.props["size"] = size
    return element


def create_native_textarea(label: str, value: str, on_change: T.Callable[[str], None], *,
                           placeholder: str = "", num_rows: int = 8,
                           num_columns: int = 80) -> ui.element:
    """Creates a labeled `<textarea>`, which works like an input of `create_native_input`."""
    element = _create_text_control("textarea", label, value, on_change, placeholder)
    element.props.update(rows=num_rows, cols=num_columns)
    return element


def _create_text_control(tag: str, label: str, value: str, on_change: T.Callable[[str], None],
                         placeholder: str) -> ui.element:
    """Creates a labeled `<input>` or `<textarea>` (see `create_native_input`)."""
    with _create_field(label):
        element = ui.element(tag).classes("native-control")
    element.props.update(defaultValue=value, placeholder=placeholder)
    # The result is returned, so that NiceGUI awaits that of an async `on_change`.
    element.on("change", lambda event: on_change(event.args),
               js_handler="(e) => emit(e.target.value)")
    return element


def create_native_text_form(control_f: T.Callable[..., ui.element], label: str, value: str,
                            button_text: str, on_saved: T.Callable[[str], T.Any],
                            **control_kwargs) -> None:
    """Creates a row of a text control and a button whose click passes the control's text to
    `on_saved`, which can be async.

    Args:
        control_f: `create_native_input` or `create_native_textarea`, which gets `control_kwargs`.
    """
    text = {"value": value}
    with ui.element("div").classes("form-row items-end"):
        control_f(label, value, lambda t: text.update(value=t), **control_kwargs)
        create_native_button(button_text, lambda: on_saved(text["value"]))


def set_native_input_placeholder(element: ui.element, placeholder: str) -> None:
    """Sets the placeholder of an input of `create_native_input`, e.g. to show the value that a
    blank input stands for. The element is rendered again only if the placeholder changes."""
    if element.props.get("placeholder") != placeholder:
        element.props["placeholder"] = placeholder
        element.update()


def create_native_checkbox(label: str, value: bool, on_change: T.Callable[[bool], None], *,
                           is_label_html: bool = False) -> ui.element:
    """
    Args:
        is_label_html: Whether `label` is HTML, e.g. with a link, which a click follows instead
            of checking the checkbox.
    """
    with ui.element("label").classes("flex items-center gap-1"):
        element = ui.element("input").classes("native-control")
        element.props.update(type="checkbox", checked=value, autocomplete="off")
        if is_label_html:
            ui.html(label, sanitize=False)
        else:
            ui.label(label)
    element.on("change", _on_user_change(element, "checked", lambda v: on_change(bool(v))),
               js_handler="(e) => emit(e.target.checked)")
    return element


def set_native_checkbox_checked(checkbox: ui.element, is_checked: bool) -> None:
    """Checks or unchecks a checkbox of `create_native_checkbox` from code, without its
    `on_change`."""
    if checkbox.props["checked"] != is_checked:
        checkbox.props["checked"] = is_checked
        checkbox.update()


def create_native_button(text: str, on_click: T.Callable[[], T.Any]) -> ui.element:
    button = TextElement(tag="button", text=text).classes("native-control")
    button.props["type"] = "button"
    button.on("click", lambda _: on_click())
    return button


def set_status(label: ui.label, text: str, is_error: bool = False,
               is_busy: bool = False) -> None:
    """Shows the status of a form in `label`, e.g. the result of a click, with a spinner if
    `is_busy`, e.g. while a file is prepared."""
    label.set_text(text)
    label.classes(replace="error" if is_error else "muted busy" if is_busy else "muted")


def create_file_upload_input(label: str, upload_url: str, accept: str,
                             on_status: T.Callable[[dict], None]) -> ui.element:
    """Creates a labeled file `<input>` that posts the chosen file to `upload_url` as the request
    body, with its name as the query parameter 'file_name'.

    Args:
        upload_url: An endpoint that stores the file and returns a JSON object.
        accept: The `accept` attribute, e.g. '.parquet'.
        on_status: Gets `{'status': 'uploading', 'file_name': ...}` when the upload starts,
            then the JSON object of the endpoint, or `{'status': 'error', 'message': ...}`.
            Each also has an 'upload_id', which increases with each upload, so that the reply
            to an earlier upload can be recognized.
    """
    with _create_field(label):
        element = ui.element("input").classes("native-control")
    element.props.update(type="file", accept=accept)
    url = json.dumps(upload_url)
    element.on("change", lambda e: on_status(e.args), js_handler=f"""(e) => {{
        const file = e.target.files[0];
        if (!file) return;
        const uploadId = e.target.uploadId = (e.target.uploadId || 0) + 1;
        emit({{status: "uploading", upload_id: uploadId, file_name: file.name}});
        fetch({url} + "?" + new URLSearchParams({{file_name: file.name}}),
              {{method: "POST", body: file}})
            .then(async (r) => r.ok ? await r.json()
                                    : {{status: "error", message: await r.text()}})
            .catch((error) => ({{status: "error", message: String(error)}}))
            .then((status) => emit({{...status, upload_id: uploadId}}));
    }}""")
    return element
