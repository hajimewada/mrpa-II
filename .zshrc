
export PATH="/Users/Hajim/.local/bin:$PATH"
# Change Terminal.app tab background color while a Python venv is active.
# Apple Terminal has no ANSI escape code for background color, so this goes
# through AppleScript. Colors are RGB triplets scaled 0-65535 (classic Mac RGBColor).
typeset -g __VENV_BG_ORIG=""
typeset -g __VENV_BG_LAST=""
__VENV_BG_ACTIVE_COLOR="2570, 5140, 15420" # dark navy blue

__venv_bg_set_color() {
  osascript -e "tell application \"Terminal\" to set background color of selected tab of front window to {$1}" >/dev/null 2>&1
}

__venv_bg_get_color() {
  osascript -e 'tell application "Terminal" to get background color of selected tab of front window' 2>/dev/null
}

__venv_bg_update() {
  if [[ -n "$VIRTUAL_ENV" ]]; then
    if [[ "$__VENV_BG_LAST" != "$VIRTUAL_ENV" ]]; then
      [[ -z "$__VENV_BG_LAST" ]] && __VENV_BG_ORIG=$(__venv_bg_get_color)
      __venv_bg_set_color "$__VENV_BG_ACTIVE_COLOR"
      __VENV_BG_LAST="$VIRTUAL_ENV"
    fi
  elif [[ -n "$__VENV_BG_LAST" ]]; then
    [[ -n "$__VENV_BG_ORIG" ]] && __venv_bg_set_color "$__VENV_BG_ORIG"
    __VENV_BG_LAST=""
    __VENV_BG_ORIG=""
  fi
}
#autoload -U add-zsh-hook
#add-zsh-hook precmd __venv_bg_update