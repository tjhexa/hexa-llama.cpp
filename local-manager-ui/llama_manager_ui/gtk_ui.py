"""GTK3 user interface for local-manager-ui.

Threading model: all subprocess work runs on background threads. Every
callback that touches a widget is dispatched through GLib.idle_add so only
the main thread builds UI.
"""

from __future__ import annotations

import html
import threading
import time
from pathlib import Path
from typing import Any, Callable

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Pango  # noqa: E402

from . import core  # noqa: E402

APP_ID = "org.llamacpp.LocalManagerUI"
MAX_LOG_LINES = 6000


def idle(fn: Callable[[], None]) -> None:
    GLib.idle_add(lambda: (fn(), False)[1])


def _mono() -> Pango.FontDescription:
    fd = Pango.FontDescription("monospace 10")
    return fd


def _field_row(label: str, widget: Gtk.Widget, tooltip: str = "") -> Gtk.Box:
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    lbl = Gtk.Label(label=label, xalign=0.0)
    lbl.set_size_request(110, -1)
    if tooltip:
        lbl.set_tooltip_text(tooltip)
        widget.set_tooltip_text(tooltip)
    row.pack_start(lbl, False, False, 0)
    row.pack_start(widget, True, True, 0)
    return row


# --------------------------------------------------------------------------
# log view
# --------------------------------------------------------------------------

class LogView(Gtk.Box):
    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.set_border_width(4)

        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.title = Gtk.Label(label="log", xalign=0.0)
        self.title.get_style_context().add_class("dim-label")
        bar.pack_start(self.title, False, False, 0)

        self.autoscroll = Gtk.CheckButton(label="auto-scroll")
        self.autoscroll.set_active(True)
        bar.pack_end(self.autoscroll, False, False, 0)
        copy_btn = Gtk.Button.new_from_icon_name("edit-copy-symbolic", Gtk.IconSize.BUTTON)
        copy_btn.set_tooltip_text("Copy all")
        copy_btn.connect("clicked", self._on_copy)
        bar.pack_end(copy_btn, False, False, 0)
        clear_btn = Gtk.Button.new_from_icon_name("edit-clear-symbolic", Gtk.IconSize.BUTTON)
        clear_btn.set_tooltip_text("Clear")
        clear_btn.connect("clicked", lambda *_: self.clear())
        bar.pack_end(clear_btn, False, False, 0)
        self.pack_start(bar, False, False, 0)

        self.buffer = Gtk.TextBuffer()
        self.buffer.create_tag("ts", foreground="#888888")
        self.buffer.create_tag("err", foreground="#e01b24")
        self.buffer.create_tag("ok", foreground="#2ec27e")
        self.buffer.create_tag("sys", foreground="#1c71d8")
        self.buffer.create_tag("warn", foreground="#e5a50a")

        self.view = Gtk.TextView(buffer=self.buffer)
        self.view.set_editable(False)
        self.view.set_cursor_visible(False)
        self.view.set_monospace(True)
        self.view.override_font(_mono())
        self.view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.add(self.view)
        self.pack_start(scroll, True, True, 0)

    def append(self, text: str, tag: str = "") -> None:
        end = self.buffer.get_end_iter()
        if tag:
            self.buffer.insert_with_tags_by_name(end, text + "\n", tag)
        else:
            self.buffer.insert(end, text + "\n")
        if self.buffer.get_line_count() > MAX_LOG_LINES:
            start = self.buffer.get_start_iter()
            cut = self.buffer.get_iter_at_line(self.buffer.get_line_count() - MAX_LOG_LINES)
            self.buffer.delete(start, cut)
        if self.autoscroll.get_active():
            mark = self.buffer.create_mark(None, self.buffer.get_end_iter(), False)
            self.view.scroll_mark_onscreen(mark)
            self.buffer.delete_mark(mark)

    def line(self, text: str, tag: str = "") -> None:
        stamp = time.strftime("%H:%M:%S")
        end = self.buffer.get_end_iter()
        self.buffer.insert_with_tags_by_name(end, f"[{stamp}] ", "ts")
        self.append(text, tag)

    def clear(self) -> None:
        self.buffer.set_text("")

    def _on_copy(self, *_: Any) -> None:
        start, end = self.buffer.get_bounds()
        cb = Gtk.Clipboard.get_default(self.get_display())
        cb.set_text(self.buffer.get_text(start, end, False), -1)
        self.buffer.select_range(start, end)


# --------------------------------------------------------------------------
# applications
# --------------------------------------------------------------------------

class LocalManagerApp(Gtk.Application):
    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__(application_id=APP_ID)
        self.cfg, self.cfg_path = core.load_config(config_path)
        self.hist = core.load_history()
        self.job: core.SpawnedJob | None = None
        self.job_kind = ""
        self.last_opts: core.LaunchOptions | None = None
        self.win: MainWindow | None = None

    def do_activate(self) -> None:  # noqa: N802
        if self.win is None:
            self.win = MainWindow(self)
        self.win.present()

    def is_busy(self) -> bool:
        return self.job is not None and self.job.running

    def start_server(self, opts: core.LaunchOptions) -> None:
        if self.win:
            self.win.start_server(opts)

    def start_build(self) -> None:
        if self.win:
            self.win.start_build()


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: LocalManagerApp) -> None:
        super().__init__(application=app, title="llama.cpp local manager")
        self.app = app
        self.cfg = app.cfg
        self.set_default_size(1180, 760)
        self.set_position(Gtk.WindowPosition.CENTER)

        self._load_css()
        self.ai = 0  # idle-load cursor for model metadata
        self._models: list[core.ModelRef] = []

        header = Gtk.HeaderBar()
        header.set_show_close_button(True)
        header.set_title("llama.cpp local manager")
        header.set_subtitle(f"{core.REPO_ROOT}")
        self.set_titlebar(header)
        self.status_label = Gtk.Label(label="idle")
        self.status_label.get_style_context().add_class("lm-status")
        header.pack_end(self.status_label)
        self.stop_btn = Gtk.Button.new_with_label("Stop")
        self.stop_btn.get_style_context().add_class("destructive-action")
        self.stop_btn.set_sensitive(False)
        self.stop_btn.set_no_show_all(True)
        self.stop_btn.connect("clicked", self._on_stop)
        header.pack_end(self.stop_btn)
        dark_btn = Gtk.ToggleButton()
        dark_btn.set_image(Gtk.Image.new_from_icon_name("weather-clear-night-symbolic", Gtk.IconSize.BUTTON))
        dark_btn.set_tooltip_text("Toggle dark theme")
        dark_btn.connect("toggled", self._on_dark_toggled)
        header.pack_start(dark_btn)

        self.stack_notebook = Gtk.Notebook()
        self.stack_notebook.connect("switch-page", self._on_tab_switch)
        self.nb = self.stack_notebook

        self.models_view = ModelsView(self)
        self.server_view = ServerView(self)
        self.build_view = BuildView(self)
        self.settings_view = SettingsView(self)

        self.nb.append_page(self.models_view, Gtk.Label(label="Models"))
        self.nb.append_page(self.server_view, Gtk.Label(label="Server"))
        self.nb.append_page(self.build_view, Gtk.Label(label="Build"))
        self.nb.append_page(self.settings_view, Gtk.Label(label="Settings"))
        self.add(self.nb)

        self.refresh_models()
        self.refresh_git()
        self._tick_idle_scan()
        self.show_all()
        self.nb.set_current_page(0)

    # -- styling ---------------------------------------------------------

    def _load_css(self) -> None:
        css = b"""
        .lm-status {
            background: alpha(currentColor, 0.08);
            border-radius: 6px;
            padding: 2px 10px;
            font-weight: bold;
        }
        .lm-card {
            background: alpha(currentColor, 0.04);
            border-radius: 8px;
            padding: 10px;
        }
        .lm-title { font-weight: bold; font-size: 1.05em; }
        .lm-mono { font-family: monospace; }
        .lm-warn { color: #e5a50a; }
        .lm-bad { color: #e01b24; }
        .lm-ok { color: #2ec27e; }
        """
        try:
            provider = Gtk.CssProvider()
            provider.load_from_data(css)
            Gtk.StyleContext.add_provider_for_screen(
                self.get_screen(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        except Exception:  # noqa: BLE001 - CSS is cosmetic
            pass

    def _on_dark_toggled(self, btn: Gtk.ToggleButton) -> None:
        settings = Gtk.Settings.get_default()
        if settings is not None:
            settings.set_property("gtk-application-prefer-dark-theme", btn.get_active())

    # -- status ----------------------------------------------------------

    def set_status(self, text: str, kind: str = "") -> None:
        self.status_label.set_text(text)
        ctx = self.status_label.get_style_context()
        for cls in ("lm-ok", "lm-bad", "lm-warn"):
            ctx.remove_class(cls)
        if kind:
            ctx.add_class(kind)

    def _refresh_stop_visibility(self) -> None:
        busy = self.app.is_busy()
        self.stop_btn.set_sensitive(busy)
        self.stop_btn.set_visible(busy)

    def _begin_job(self, kind: str) -> bool:
        if self.app.is_busy():
            self._warn("Another job is already running. Stop it first.")
            return False
        self.app.job_kind = kind
        self._refresh_stop_visibility()
        return True

    def _warn(self, msg: str) -> None:
        dialog = Gtk.MessageDialog(
            transient_for=self, flags=0, message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK, text=msg,
        )
        dialog.run()
        dialog.destroy()

    def _on_stop(self, *_: Any) -> None:
        if self.app.job and self.app.job.running:
            self.set_status("stopping...", "lm-warn")
            target = self.app.job
            threading.Thread(target=target.stop, daemon=True).start()

    # -- model list ------------------------------------------------------

    def refresh_models(self) -> None:
        models_dir = Path(self.cfg.get("models_dir", "")).expanduser()
        self._models = core.discover_models(models_dir)
        self.app.ai = 0
        self.models_view.set_models(self._models, last_model=self.app.hist.get("last_model", ""))
        self.server_view.set_models(self._models, last_model=self.app.hist.get("last_model", ""))
        self.set_status(f"{len(self._models)} models", "")
        if not self._models:
            self.models_view.show_empty(str(models_dir))
    def _tick_idle_scan(self) -> bool:
        """Lazily read GGUF headers a few at a time so the UI stays instant."""
        if self.app.ai < len(self._models):
            end = min(self.app.ai + 3, len(self._models))
            for i in range(self.app.ai, end):
                self.models_view.load_row_metadata(i)
            self.app.ai = end
            return True
        return False

    def refresh_git(self) -> None:
        info = core.git_info()
        branch = info.get("branch", "?")
        head = info.get("head", "?")
        dirty = " *" if info.get("dirty") else ""
        self.build_view.set_git_info(f"{branch} @{head}{dirty}")
        if self.server_view and self.app.job is None:
            self.set_status(f"{branch}{dirty}", "")

    # -- launch ----------------------------------------------------------

    def start_server(self, opts: core.LaunchOptions) -> None:
        if not self._begin_job("server"):
            return
        url = core.server_url(opts)
        if core.port_in_use(opts.host, opts.port):
            self.server_view.log.line(f"Port {opts.port} already in use -> {url}", "warn")
            self.server_view.log.line("Opening the existing server instead.", "sys")
            core.open_in_browser(url)
            return

        cmd = core.build_command(opts)
        exe = Path(cmd[0])
        if not exe.exists():
            self._warn(f"Binary not found:\n{exe}\n\nBuild llama.cpp or fix bin_dir in Settings.")
            return

        self.last_opts = opts
        self.app.hist = core.record_launch(self.app.hist, opts)
        try:
            core.save_history(self.app.hist)
        except OSError:
            pass
        self.server_view.refresh_recent()

        log = self.server_view.log
        log.line(f"$ {core.command_to_shell(cmd)}", "sys")
        self.set_status("server starting...", "lm-warn")
        self.server_view.set_running(True, url)

        self.app.job = core.spawn_job(
            cmd, mode="server", label="llama-server", cwd=core.REPO_ROOT,
            on_line=lambda line: log.append(core.strip_ansi(line)),
            on_done=self._on_server_done,
            idle_scheduler=idle,
        )
        self._refresh_stop_visibility()
        self.server_view.start_health_probe(url, open_browser=bool(self.cfg.get("open_browser", True)))

    def _on_server_done(self, code: int) -> None:
        self.app.job = None
        self._refresh_stop_visibility()
        self.server_view.set_running(False, "")
        if code == 0:
            self.server_view.log.line("server exited cleanly", "ok")
            self.set_status("idle", "")
        else:
            self.server_view.log.line(f"server exited with code {code}", "err")
            self.set_status(f"server exit {code}", "lm-bad")

    # -- CLI -------------------------------------------------------------

    def run_cli(self, opts: core.LaunchOptions) -> None:
        binary = core.find_cli_bin(opts.bin_dir, mmproj=bool(opts.model.mmproj and opts.use_mmproj),
                                   preferred=opts.cli_bin)
        if binary is None:
            self._warn(f"No CLI binary found in {opts.bin_dir}.")
            return
        opts.cli_bin = binary.name
        cmd = core.build_command(opts)
        if not binary.exists():
            self._warn(f"Binary not found:\n{binary}")
            return

        self.app.hist = core.record_launch(self.app.hist, opts)
        try:
            core.save_history(self.app.hist)
        except OSError:
            pass

        term = core.find_terminal()
        log = self.server_view.log
        if term is None:
            self._warn("No terminal emulator found (gnome-terminal, konsole, ...).")
            log.line("CLI launch needs a terminal emulator.", "err")
            return
        name, base = term
        log.line(f"$ {core.command_to_shell(cmd)}", "sys")
        log.line(f"launching in {name}...", "sys")
        core.spawn_job(base + cmd, mode="cli", label=opts.model.name, cwd=core.REPO_ROOT,
                       on_line=lambda line: log.append(core.strip_ansi(line)),
                       idle_scheduler=idle)

    # -- build -----------------------------------------------------------

    def start_build(self) -> None:
        if not self._begin_job("build"):
            return
        spec = self.build_view.build_spec()
        if not core.BUILD_SCRIPT.exists():
            self._warn(f"Build script not found:\n{core.BUILD_SCRIPT}")
            return
        cmd, env = core.build_script_cmd(**spec)
        log = self.build_view.log
        log.clear()
        log.line(f"$ {core.command_to_shell(cmd)}", "sys")
        if env:
            log.line("env: " + " ".join(f"{k}={v}" for k, v in env.items()), "sys")
        self.set_status("building...", "lm-warn")
        self.build_view.set_running(True)

        def on_line(line: str) -> None:
            clean = core.strip_ansi(line)
            low = clean.lower()
            tag = ""
            if " error" in low or "error:" in low or clean.startswith("!!") or "failed" in low:
                tag = "err"
            elif clean.startswith("==>") or clean.startswith("-->"):
                tag = "sys"
            log.append(clean, tag)

        self.app.job = core.spawn_job(
            cmd, mode="build", label="pull-merge-build.sh", cwd=core.REPO_ROOT, env=env,
            on_line=on_line, on_done=self._on_build_done, idle_scheduler=idle,
        )
        self._refresh_stop_visibility()

    def _on_build_done(self, code: int) -> None:
        self.app.job = None
        self._refresh_stop_visibility()
        self.build_view.set_running(False)
        if code == 0:
            self.build_view.log.line("build finished OK", "ok")
            self.set_status("build OK", "lm-ok")
        else:
            self.build_view.log.line(f"build failed (exit {code})", "err")
            self.set_status(f"build exit {code}", "lm-bad")
        self.refresh_git()

    # -- misc ------------------------------------------------------------

    def _on_tab_switch(self, _nb: Gtk.Notebook, page: Gtk.Widget, _idx: int) -> None:
        if page is self.build_view:
            self.refresh_git()

    def open_url(self, url: str) -> None:
        if not core.open_in_browser(url):
            self._warn(f"Could not open a browser. URL:\n{url}")


# --------------------------------------------------------------------------
# models tab
# --------------------------------------------------------------------------

class ModelsView(Gtk.Box):
    COLS = ("Name", "Quant", "Size", "Ctx", "Est. VRAM")

    def __init__(self, win: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.win = win
        self.set_border_width(8)
        self.set_margin_top(4)
        self._models: list[core.ModelRef] = []
        self._last_model = ""
        self._selected_model: core.ModelRef | None = None

        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.filter = Gtk.SearchEntry()
        self.filter.set_placeholder_text("Filter models (name, quant, arch)")
        self.filter.connect("search-changed", self._on_filter)
        top.pack_start(self.filter, True, True, 0)
        refresh = Gtk.Button.new_from_icon_name("view-refresh-symbolic", Gtk.IconSize.BUTTON)
        refresh.set_tooltip_text("Rescan models_dir")
        refresh.connect("clicked", lambda *_: self.win.refresh_models())
        top.pack_start(refresh, False, False, 0)
        self.pack_start(top, False, False, 0)

        self.store = Gtk.ListStore(str, str, str, str, str, object)
        self.filtered = self.store.filter_new()
        self.filtered.set_visible_func(self._visible_func)
        self.tree = Gtk.TreeView(model=self.filtered)
        self.tree.set_tooltip_column(0)
        for i, title in enumerate(self.COLS):
            col = Gtk.TreeViewColumn(title, Gtk.CellRendererText(), text=i)
            if title in ("Size", "Ctx", "Est. VRAM"):
                col.set_alignment(1.0)
                for rend in col.get_cells():
                    rend.set_property("xalign", 1.0)
            col.set_resizable(True)
            col.set_sort_column_id(i)
            self.tree.append_column(col)
        self.tree.set_search_column(0)
        sel = self.tree.get_selection()
        sel.connect("changed", self._on_select)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)
        scroll.add(self.tree)
        self.pack_start(scroll, True, True, 0)

        details = Gtk.Frame(label="Model details")
        self.detail_buf = Gtk.TextBuffer()
        self.detail = Gtk.TextView(buffer=self.detail_buf)
        self.detail.set_editable(False)
        self.detail.set_monospace(True)
        self.detail.override_font(_mono())
        self.detail.set_size_request(-1, 150)
        dscroll = Gtk.ScrolledWindow()
        dscroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        dscroll.add(self.detail)
        details.add(dscroll)
        self.pack_start(details, False, False, 0)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        load_btn = Gtk.Button(label="Load into Server tab")
        load_btn.connect("clicked", self._on_load_into_server)
        actions.pack_start(load_btn, False, False, 0)
        web_btn = Gtk.Button(label="Launch web")
        web_btn.get_style_context().add_class("suggested-action")
        web_btn.connect("clicked", self._on_launch_web)
        actions.pack_start(web_btn, False, False, 0)
        cli_btn = Gtk.Button(label="Launch CLI")
        cli_btn.connect("clicked", self._on_launch_cli)
        actions.pack_start(cli_btn, False, False, 0)
        copy_btn = Gtk.Button(label="Copy path")
        copy_btn.connect("clicked", self._on_copy_path)
        actions.pack_end(copy_btn, False, False, 0)
        self.pack_start(actions, False, False, 0)

    # -- model list wiring ----------------------------------------------

    def set_models(self, models: list[core.ModelRef], last_model: str = "") -> None:
        self._models = models
        self._last_model = last_model
        self.store.clear()
        for m in models:
            marker = "  (last)" if str(m.path) == last_model else ""
            self.store.append([m.name + marker, "", "", "", "", m])
        if models:
            self.tree.get_selection().select_iter(self.filtered.get_iter_first())

    def load_row_metadata(self, index: int) -> None:
        if index >= len(self.store):
            return
        it = self.store.get_iter(Gtk.TreePath.new_from_indices([index]))
        model: core.ModelRef = self.store.get_value(it, 5)
        if model is None or self.store.get_value(it, 1):
            return
        info = core.read_gguf_info(model.path)
        model._info = info  # type: ignore[attr-defined]
        ctx = str(info.context_length) if info.context_length else "-"
        est = core.estimate_vram_mib(info.size_bytes, info.context_length, info.block_count)
        self.store.set(it, 1, info.quant, 2, info.size_human, 3, ctx, 4, f"{est / 1024.0:.1f} GiB")
        if self._selected_model is model:
            self._show_details(model)

    def show_empty(self, models_dir: str) -> None:
        self.detail_buf.set_text(f"No .gguf files found under:\n  {models_dir}\n\nSet models_dir in the Settings tab.")

    def _selected_model_by_path(self, path: str) -> core.ModelRef | None:
        for m in self._models:
            if str(m.path) == path:
                return m
        return None

    # -- helpers ---------------------------------------------------------

    def _visible_func(self, model: Gtk.TreeModel, it: Gtk.TreeIter, _data: Any) -> bool:
        text = self.filter.get_text().strip().lower()
        if not text:
            return True
        hay = " ".join(str(model.get_value(it, i)) for i in range(5)).lower()
        return text in hay

    def _on_filter(self, *_: Any) -> None:
        self.filtered.refilter()

    def _on_select(self, sel: Gtk.TreeSelection) -> None:
        model, it = sel.get_selected()
        if model is None or it is None:
            return
        child = self.filtered.convert_iter_to_child_iter(it)
        self._selected_model = self.store.get_value(child, 5)
        if self._selected_model is not None:
            self._show_details(self._selected_model)

    def _show_details(self, model: core.ModelRef) -> None:
        info = getattr(model, "_info", None)
        if info is None:
            self.detail_buf.set_text("Reading GGUF metadata...")
            return
        lines = [
            f"name        : {info.display_name}",
            f"path        : {model.path}",
            f"architecture: {info.architecture or '?'}",
            f"quant       : {info.quant}",
            f"size        : {info.size_human}",
            f"max context : {info.context_length or '?'}",
            f"emb. length : {info.embedding_length or '?'}",
            f"blocks      : {info.block_count or '?'}",
            f"rope base   : {info.rope_freq_base or '?'}",
            f"rope scaling: {info.rope_scaling_type or '-'} "
            f"(factor {info.rope_scaling_factor or '-'}, orig ctx {info.original_context_length or '-'})",
            f"experts     : {info.expert_used_count or '-'}/{info.expert_count or '-'}",
            f"mmproj      : {model.mmproj or '-'}",
        ]
        if info.error:
            lines.append(f"error       : {info.error}")
        self.detail_buf.set_text("\n".join(lines))

    # -- actions ---------------------------------------------------------

    def _opts_from_form(self, mode: str) -> core.LaunchOptions:
        model = self._selected_model
        if model is None:
            raise ValueError("no model selected")
        return core.LaunchOptions(
            mode=mode,
            model=model,
            bin_dir=Path(self.win.cfg.get("bin_dir", "")).expanduser(),
            server_bin=str(self.win.cfg.get("server_bin", "llama-server")),
            cli_bin=str(self.win.cfg.get("cli_bin", "llama-completion")),
            host=str(self.win.cfg.get("default_host", "127.0.0.1")),
            port=int(self.win.cfg.get("default_port", 9898) or 9898),
            ngl=int(self.win.cfg.get("default_ngl", 99) or 99),
            ctx=int(self.win.cfg.get("default_ctx", 0) or 0),
            alias=str(self.win.cfg.get("alias", "")),
            api_key=str(self.win.cfg.get("api_key", "")),
            use_mmproj=not bool(self.win.cfg.get("skip_mmproj", False)),
            extra_args=[str(x) for x in (self.win.cfg.get("extra_server_args") or [])],
        )

    def _on_launch_web(self, *_: Any) -> None:
        try:
            opts = self._opts_from_form("web")
        except ValueError:
            self.win._warn("Select a model first.")
            return
        self.win.server_view.apply_options(opts)
        self.win.nb.set_current_page(1)
        self.win.start_server(opts)

    def _on_launch_cli(self, *_: Any) -> None:
        try:
            opts = self._opts_from_form("cli")
        except ValueError:
            self.win._warn("Select a model first.")
            return
        opts.extra_args = [str(x) for x in (self.win.cfg.get("extra_cli_args") or [])]
        self.win.nb.set_current_page(1)
        self.win.run_cli(opts)

    def _on_load_into_server(self, *_: Any) -> None:
        model = self._selected_model
        if model is None:
            self.win._warn("Select a model first.")
            return
        self.win.server_view.select_model_path(str(model.path))
        self.win.nb.set_current_page(1)

    def _on_copy_path(self, *_: Any) -> None:
        model = self._selected_model
        if model is None:
            return
        cb = Gtk.Clipboard.get_default(self.get_display())
        cb.set_text(str(model.path), -1)


# --------------------------------------------------------------------------
# server tab
# --------------------------------------------------------------------------

class ServerView(Gtk.Box):
    def __init__(self, win: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win
        self._probe_source: int | None = None

        pane = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        pane.set_position(380)

        # left: controls
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        left.set_border_width(10)
        title = Gtk.Label(label="Launch options", xalign=0.0)
        title.get_style_context().add_class("lm-title")
        left.pack_start(title, False, False, 0)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.connect("changed", self._on_model_changed)
        left.pack_start(_field_row("Model", self.model_combo), False, False, 0)

        self.mode_combo = Gtk.ComboBoxText()
        self.mode_combo.append_text("server")
        self.mode_combo.append_text("cli")
        self.mode_combo.set_active(0)
        self.mode_combo.connect("changed", lambda *_: self._update_preview())
        left.pack_start(_field_row("Mode", self.mode_combo), False, False, 0)

        self.ngl = Gtk.SpinButton.new_with_range(0, 999, 1)
        left.pack_start(_field_row("GPU layers -ngl", self.ngl, "0 = CPU only"), False, False, 0)

        self.ctx = Gtk.SpinButton.new_with_range(0, 10_000_000, 512)
        left.pack_start(_field_row("Context -c", self.ctx, "0 = model default"), False, False, 0)

        self.host = Gtk.Entry()
        left.pack_start(_field_row("Host", self.host), False, False, 0)

        self.port = Gtk.SpinButton.new_with_range(1, 65535, 1)
        left.pack_start(_field_row("Port", self.port), False, False, 0)

        self.alias = Gtk.Entry()
        left.pack_start(_field_row("API alias -a", self.alias), False, False, 0)

        self.api_key = Gtk.Entry()
        self.api_key.set_visibility(False)
        self.api_key.set_placeholder_text("optional")
        left.pack_start(_field_row("API key", self.api_key), False, False, 0)

        self.mmproj = Gtk.CheckButton(label="Attach sibling mmproj")
        left.pack_start(self.mmproj, False, False, 0)

        self.extra = Gtk.Entry()
        self.extra.set_placeholder_text("extra args, space separated")
        left.pack_start(_field_row("Extra args", self.extra), False, False, 0)

        self.cmd_preview = Gtk.Label(xalign=0.0)
        self.cmd_preview.set_line_wrap(True)
        self.cmd_preview.set_selectable(True)
        self.cmd_preview.get_style_context().add_class("dim-label")
        self.cmd_preview.get_style_context().add_class("lm-mono")
        for w in (self.ngl, self.ctx, self.host, self.port, self.alias, self.extra):
            w.connect("changed", lambda *_: self._update_preview())
        self.mmproj.connect("toggled", lambda *_: self._update_preview())
        left.pack_start(Gtk.Separator(), False, False, 4)
        left.pack_start(self.cmd_preview, False, False, 0)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.start_btn = Gtk.Button(label="Start")
        self.start_btn.get_style_context().add_class("suggested-action")
        self.start_btn.connect("clicked", self._on_start)
        btns.pack_start(self.start_btn, True, True, 0)
        cli_btn = Gtk.Button(label="Start CLI")
        cli_btn.connect("clicked", self._on_start_cli)
        btns.pack_start(cli_btn, True, True, 0)
        left.pack_start(btns, False, False, 0)

        self.open_btn = Gtk.LinkButton.new_with_label("#", "Open web UI")
        self.open_btn.set_sensitive(False)
        self.open_btn.connect("clicked", self._on_open_clicked)
        left.pack_start(self.open_btn, False, False, 0)

        recent_title = Gtk.Label(label="Recent", xalign=0.0)
        recent_title.get_style_context().add_class("dim-label")
        left.pack_start(recent_title, False, False, 6)
        self.recent_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        left.pack_start(self.recent_box, False, False, 0)
        self.refresh_recent()

        pane.pack1(left, resize=True, shrink=False)

        # right: log
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        right.set_border_width(6)
        self.banner = Gtk.Label(xalign=0.0)
        self.banner.get_style_context().add_class("lm-card")
        self.banner.set_line_wrap(True)
        right.pack_start(self.banner, False, False, 0)
        self.log = LogView()
        right.pack_start(self.log, True, True, 0)
        pane.pack2(right, resize=True, shrink=False)

        self.pack_start(pane, True, True, 0)
        self.set_running(False, "")
        self.seed_defaults()

    def seed_defaults(self) -> None:
        cfg = self.win.cfg
        self.ngl.set_value(int(cfg.get("default_ngl", 99) or 99))
        self.ctx.set_value(int(cfg.get("default_ctx", 0) or 0))
        self.host.set_text(str(cfg.get("default_host", "127.0.0.1")))
        self.port.set_value(int(cfg.get("default_port", 9898) or 9898))
        self.alias.set_text(str(cfg.get("alias", "")))
        self.api_key.set_text(str(cfg.get("api_key", "") or ""))
        self.mmproj.set_active(not bool(cfg.get("skip_mmproj", False)))
        self._update_preview()

    # -- model chooser ---------------------------------------------------

    def set_models(self, models: list[core.ModelRef], last_model: str = "") -> None:
        self.model_combo.remove_all()
        self._paths = [str(m.path) for m in models]
        for m in models:
            self.model_combo.append_text(m.name)
        if models:
            idx = self._paths.index(last_model) if last_model in self._paths else 0
            self.model_combo.set_active(idx)

    def select_model_path(self, path: str) -> None:
        idx = self.model_combo.get_model()
        for i, p in enumerate(self._paths):
            if p == path:
                self.model_combo.set_active(i)
                return

    def _on_model_changed(self, *_: Any) -> None:
        model = self._current_model()
        if model is None:
            return
        info = getattr(model, "_info", None)
        if not self.alias.get_text().strip():
            self.alias.set_placeholder_text(model.path.stem)
        self._update_preview()
        if info and info.context_length:
            self.ctx.set_tooltip_text(f"model max context = {info.context_length}")

    def _current_model(self) -> core.ModelRef | None:
        active = self.model_combo.get_active()
        if active < 0 or active >= len(self._paths):
            return None
        return self.win.models_view._selected_model_by_path(self._paths[active])

    # -- options ---------------------------------------------------------

    def apply_options(self, opts: core.LaunchOptions) -> None:
        self.select_model_path(str(opts.model.path))
        self.mode_combo.set_active(0 if opts.mode == "web" else 1)
        self.ngl.set_value(opts.ngl)
        self.ctx.set_value(opts.ctx)
        self.host.set_text(opts.host)
        self.port.set_value(opts.port)
        self.alias.set_text(opts.alias)
        self.api_key.set_text(opts.api_key or "")
        self.mmproj.set_active(opts.use_mmproj)
        self._update_preview()

    def _collect_opts(self, mode: str) -> core.LaunchOptions | None:
        model = self._current_model()
        if model is None:
            return None
        extra = self.extra.get_text().split()
        return core.LaunchOptions(
            mode=mode,
            model=model,
            bin_dir=Path(self.win.cfg.get("bin_dir", "")).expanduser(),
            server_bin=str(self.win.cfg.get("server_bin", "llama-server")),
            cli_bin=str(self.win.cfg.get("cli_bin", "llama-completion")),
            host=self.host.get_text().strip() or "127.0.0.1",
            port=int(self.port.get_value()),
            ngl=int(self.ngl.get_value()),
            ctx=int(self.ctx.get_value()),
            alias=self.alias.get_text().strip(),
            api_key=self.api_key.get_text().strip(),
            use_mmproj=self.mmproj.get_active(),
            extra_args=[str(x) for x in extra],
        )

    def _update_preview(self) -> None:
        opts = self._collect_opts("web" if self.mode_combo.get_active() == 0 else "cli")
        if opts is None:
            self.cmd_preview.set_text("")
            return
        try:
            self.cmd_preview.set_text(core.command_to_shell(core.build_command(opts)))
        except Exception as exc:  # noqa: BLE001
            self.cmd_preview.set_text(f"(preview error: {exc})")

    # -- actions ---------------------------------------------------------

    def _on_start(self, *_: Any) -> None:
        opts = self._collect_opts("web")
        if opts is None:
            self.win._warn("No model selected.")
            return
        self.win.start_server(opts)

    def _on_start_cli(self, *_: Any) -> None:
        opts = self._collect_opts("cli")
        if opts is None:
            self.win._warn("No model selected.")
            return
        self.win.run_cli(opts)

    def _on_open_clicked(self, *_: Any) -> None:
        self.win.open_url(self.open_btn.get_uri())

    # -- status ----------------------------------------------------------

    def set_running(self, running: bool, url: str) -> None:
        self.start_btn.set_sensitive(not running)
        if running:
            self.banner.set_markup(f"<b>Starting server...</b> {html.escape(url)}")
        elif url:
            self.banner.set_markup(f"<b>Ready</b> {html.escape(url)}")
        else:
            self.banner.set_text("Idle. Press Start to launch llama-server.")

    def start_health_probe(self, url: str, *, open_browser: bool) -> None:
        if self._probe_source is not None:
            GLib.source_remove(self._probe_source)
        self._opened = False
        deadline = time.time() + 300

        def tick() -> bool:
            if time.time() > deadline:
                self._probe_source = None
                self.banner.set_markup("<b>Timed out waiting for server</b>")
                return False
            reachable, status = core.probe_health(url)
            if reachable and status == "ok":
                self._probe_source = None
                self.set_running(False, url)
                self.open_btn.set_uri(url)
                self.open_btn.set_sensitive(True)
                self.win.set_status("server ready", "lm-ok")
                self.log.line(f"server ready: {url}  (/v1 for OpenAI API)", "ok")
                if open_browser and not self._opened:
                    self._opened = True
                    self.win.open_url(url)
                return False
            label = "loading model..." if status in ("loading", "down") else status
            self.banner.set_markup(f"<b>Starting server...</b> {html.escape(url)} ({html.escape(label)})")
            return True

        self._probe_source = GLib.timeout_add_seconds(1, tick)

    # -- recent history --------------------------------------------------

    def refresh_recent(self) -> None:
        for child in self.recent_box.get_children():
            child.destroy()
        seen: set[str] = set()
        for item in (self.win.app.hist.get("launches") or [])[:6]:
            path = item.get("model", "")
            if not path or path in seen:
                continue
            seen.add(path)
            name = Path(path).name
            btn = Gtk.Button(label=f"{name}  ({item.get('mode', '?')})")
            btn.set_relief(Gtk.ReliefStyle.NONE)
            btn.set_halign(Gtk.Align.START)
            btn.set_tooltip_text(path)
            btn.connect("clicked", lambda _b, p=path: self._launch_recent(p))
            self.recent_box.pack_start(btn, False, False, 0)
        if not seen:
            lbl = Gtk.Label(label="(no history yet)", xalign=0.0)
            lbl.get_style_context().add_class("dim-label")
            self.recent_box.pack_start(lbl, False, False, 0)
        self.recent_box.show_all()

    def _launch_recent(self, path: str) -> None:
        self.select_model_path(path)
        self._on_start()


# --------------------------------------------------------------------------
# build tab
# --------------------------------------------------------------------------

class BuildView(Gtk.Box):
    def __init__(self, win: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win

        pane = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        pane.set_position(380)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        left.set_border_width(10)
        title = Gtk.Label(label="pull-merge-build.sh", xalign=0.0)
        title.get_style_context().add_class("lm-title")
        left.pack_start(title, False, False, 0)

        self.git_label = Gtk.Label(xalign=0.0)
        self.git_label.set_line_wrap(True)
        self.git_label.get_style_context().add_class("dim-label")
        left.pack_start(self.git_label, False, False, 0)

        self.upstream = Gtk.Entry()
        self.upstream.set_placeholder_text("master")
        left.pack_start(_field_row("Upstream branch", self.upstream, "default: master"), False, False, 0)

        self.target = Gtk.Entry()
        self.target.set_placeholder_text("(current branch)")
        left.pack_start(_field_row("Target branch", self.target, "default: script default"), False, False, 0)

        self.cuda_arch = Gtk.Entry()
        self.cuda_arch.set_placeholder_text("120")
        left.pack_start(_field_row("CUDA arch", self.cuda_arch, "default: 120"), False, False, 0)

        self.jobs = Gtk.SpinButton.new_with_range(0, 256, 1)
        self.jobs.set_tooltip_text("0 = nproc")
        left.pack_start(_field_row("Jobs -j", self.jobs), False, False, 0)

        self.build_dir = Gtk.Entry()
        self.build_dir.set_placeholder_text("build")
        left.pack_start(_field_row("Build dir", self.build_dir), False, False, 0)

        self.clean = Gtk.CheckButton(label="--clean  (full rebuild)")
        left.pack_start(self.clean, False, False, 0)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.build_btn = Gtk.Button(label="Pull, merge & build")
        self.build_btn.get_style_context().add_class("suggested-action")
        self.build_btn.connect("clicked", lambda *_: self.win.start_build())
        btns.pack_start(self.build_btn, True, True, 0)
        open_term = Gtk.Button(label="Open in terminal")
        open_term.connect("clicked", self._on_open_terminal)
        btns.pack_start(open_term, True, True, 0)
        left.pack_start(btns, False, False, 0)

        note = Gtk.Label(xalign=0.0)
        note.set_line_wrap(True)
        note.get_style_context().add_class("dim-label")
        note.set_text("Uses local-manager/pull-merge-build.sh. Stop sends SIGTERM to the script and its children.")
        left.pack_start(note, False, False, 0)
        pane.pack1(left, resize=False, shrink=False)

        self.log = LogView()
        self.log.title.set_text("build output")
        pane.pack2(self.log, resize=True, shrink=False)

        self.pack_start(pane, True, True, 0)

    def build_spec(self) -> dict[str, Any]:
        return {
            "clean": self.clean.get_active(),
            "upstream_branch": self.upstream.get_text().strip(),
            "target_branch": self.target.get_text().strip(),
            "cuda_arch": self.cuda_arch.get_text().strip(),
            "jobs": int(self.jobs.get_value()),
            "build_dir": self.build_dir.get_text().strip(),
        }

    def set_git_info(self, text: str) -> None:
        self.git_label.set_text(f"repo: {text}")

    def set_running(self, running: bool) -> None:
        self.build_btn.set_sensitive(not running)

    def _on_open_terminal(self, *_: Any) -> None:
        term = core.find_terminal()
        if term is None:
            self.win._warn("No terminal emulator found.")
            return
        _name, base = term
        spec = self.build_spec()
        cmd, env = core.build_script_cmd(**spec)
        env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
        script = f"cd {core.REPO_ROOT} && {env_prefix} {core.command_to_shell(cmd)}; exec bash"
        core.spawn_job(base + ["bash", "-lc", script], mode="term", label="build",
                       on_line=lambda _l: None, idle_scheduler=idle)


# --------------------------------------------------------------------------
# settings tab
# --------------------------------------------------------------------------

class SettingsView(Gtk.Box):
    def __init__(self, win: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.win = win
        self.set_border_width(12)

        title = Gtk.Label(label="Settings", xalign=0.0)
        title.get_style_context().add_class("lm-title")
        self.pack_start(title, False, False, 0)

        grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        cfg = win.cfg

        self.models_dir = Gtk.Entry()
        self.models_dir.set_text(str(cfg.get("models_dir", "")))
        grid.pack_start(_field_row("models_dir", self._with_browse(self.models_dir)), False, False, 0)

        self.bin_dir = Gtk.Entry()
        self.bin_dir.set_text(str(cfg.get("bin_dir", "")))
        grid.pack_start(_field_row("bin_dir", self._with_browse(self.bin_dir)), False, False, 0)

        self.default_host = Gtk.Entry()
        self.default_host.set_text(str(cfg.get("default_host", "127.0.0.1")))
        grid.pack_start(_field_row("default_host", self.default_host), False, False, 0)

        self.default_port = Gtk.SpinButton.new_with_range(1, 65535, 1)
        self.default_port.set_value(int(cfg.get("default_port", 9898) or 9898))
        grid.pack_start(_field_row("default_port", self.default_port), False, False, 0)

        self.default_ngl = Gtk.SpinButton.new_with_range(0, 999, 1)
        self.default_ngl.set_value(int(cfg.get("default_ngl", 99) or 99))
        grid.pack_start(_field_row("default_ngl", self.default_ngl), False, False, 0)

        self.default_ctx = Gtk.SpinButton.new_with_range(0, 10_000_000, 512)
        self.default_ctx.set_value(int(cfg.get("default_ctx", 0) or 0))
        grid.pack_start(_field_row("default_ctx", self.default_ctx), False, False, 0)

        self.open_browser = Gtk.CheckButton(label="Open browser when server is ready")
        self.open_browser.set_active(bool(cfg.get("open_browser", True)))
        grid.pack_start(self.open_browser, False, False, 0)

        self.skip_mmproj = Gtk.CheckButton(label="skip_mmproj")
        self.skip_mmproj.set_active(bool(cfg.get("skip_mmproj", False)))
        grid.pack_start(self.skip_mmproj, False, False, 0)

        self.extra_server = Gtk.Entry()
        self.extra_server.set_text(" ".join(str(x) for x in (cfg.get("extra_server_args") or [])))
        grid.pack_start(_field_row("extra_server_args", self.extra_server), False, False, 0)

        self.extra_cli = Gtk.Entry()
        self.extra_cli.set_text(" ".join(str(x) for x in (cfg.get("extra_cli_args") or [])))
        grid.pack_start(_field_row("extra_cli_args", self.extra_cli), False, False, 0)

        self.pack_start(grid, False, False, 0)

        self.path_label = Gtk.Label(xalign=0.0)
        self.path_label.set_line_wrap(True)
        self.path_label.get_style_context().add_class("dim-label")
        self.path_label.set_text(f"config file: {win.app.cfg_path}")
        self.pack_start(self.path_label, False, False, 6)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        save = Gtk.Button(label="Save settings")
        save.get_style_context().add_class("suggested-action")
        save.connect("clicked", self._on_save)
        btns.pack_start(save, False, False, 0)
        reload_btn = Gtk.Button(label="Reload from disk")
        reload_btn.connect("clicked", self._on_reload)
        btns.pack_start(reload_btn, False, False, 0)
        self.pack_start(btns, False, False, 0)

        self.status = Gtk.Label(xalign=0.0)
        self.status.get_style_context().add_class("dim-label")
        self.pack_start(self.status, False, False, 0)

        info = Gtk.Label(xalign=0.0)
        info.get_style_context().add_class("dim-label")
        info.set_line_wrap(True)
        info.set_text(f"repo: {core.REPO_ROOT}\nbuild script: {core.BUILD_SCRIPT}")
        self.pack_start(info, False, False, 8)

    def _with_browse(self, entry: Gtk.Entry) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        box.pack_start(entry, True, True, 0)
        btn = Gtk.Button.new_from_icon_name("folder-open-symbolic", Gtk.IconSize.BUTTON)
        btn.connect("clicked", self._on_browse, entry)
        box.pack_start(btn, False, False, 0)
        return box

    def _on_browse(self, _btn: Gtk.Button, entry: Gtk.Entry) -> None:
        dialog = Gtk.FileChooserDialog(
            title="Select folder", transient_for=self.win,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        current = entry.get_text().strip()
        if current and Path(current).is_dir():
            dialog.set_current_folder(current)
        if dialog.run() == Gtk.ResponseType.OK:
            entry.set_text(dialog.get_filename())
        dialog.destroy()

    def _on_save(self, *_: Any) -> None:
        cfg = dict(self.win.app.cfg)
        cfg.update({
            "models_dir": self.models_dir.get_text().strip(),
            "bin_dir": self.bin_dir.get_text().strip(),
            "default_host": self.default_host.get_text().strip() or "127.0.0.1",
            "default_port": int(self.default_port.get_value()),
            "default_ngl": int(self.default_ngl.get_value()),
            "default_ctx": int(self.default_ctx.get_value()),
            "open_browser": self.open_browser.get_active(),
            "skip_mmproj": self.skip_mmproj.get_active(),
            "extra_server_args": self.extra_server.get_text().split(),
            "extra_cli_args": self.extra_cli.get_text().split(),
        })
        self.win.app.cfg = cfg
        self.win.cfg = cfg
        try:
            path = core.save_config(cfg)
        except OSError as exc:
            self.status.set_text(f"save failed: {exc}")
            return
        self.status.set_text(f"saved -> {path}")
        self.win.set_status("settings saved", "lm-ok")
        self.win.refresh_models()

    def _on_reload(self, *_: Any) -> None:
        cfg, path = core.load_config()
        self.win.app.cfg = cfg
        self.win.cfg = cfg
        self.win.server_view.apply_options  # keep ref
        # repopulate
        self.models_dir.set_text(str(cfg.get("models_dir", "")))
        self.bin_dir.set_text(str(cfg.get("bin_dir", "")))
        self.default_host.set_text(str(cfg.get("default_host", "127.0.0.1")))
        self.default_port.set_value(int(cfg.get("default_port", 9898) or 9898))
        self.default_ngl.set_value(int(cfg.get("default_ngl", 99) or 99))
        self.default_ctx.set_value(int(cfg.get("default_ctx", 0) or 0))
        self.open_browser.set_active(bool(cfg.get("open_browser", True)))
        self.skip_mmproj.set_active(bool(cfg.get("skip_mmproj", False)))
        self.extra_server.set_text(" ".join(str(x) for x in (cfg.get("extra_server_args") or [])))
        self.extra_cli.set_text(" ".join(str(x) for x in (cfg.get("extra_cli_args") or [])))
        self.status.set_text(f"reloaded from {path}")
        self.win.refresh_models()


def run(config_path: Path | None = None) -> int:
    app = LocalManagerApp(config_path=config_path)
    return app.run(None)
