package com.gatherion.hidbridge

/**
 * USB HID Usage IDs (boot keyboard) for Gatherion skill / nav keys.
 * Report: [modifier, reserved, key1..key6]
 */
object HidKeyMap {
    const val MOD_NONE = 0
    const val MOD_LCTRL = 0x01
    const val MOD_LSHIFT = 0x02
    const val MOD_LALT = 0x04
    const val MOD_LGUI = 0x08

    private val usage: Map<String, Int> = mapOf(
        "a" to 0x04, "b" to 0x05, "c" to 0x06, "d" to 0x07,
        "e" to 0x08, "f" to 0x09, "g" to 0x0A, "h" to 0x0B,
        "i" to 0x0C, "j" to 0x0D, "k" to 0x0E, "l" to 0x0F,
        "m" to 0x10, "n" to 0x11, "o" to 0x12, "p" to 0x13,
        "q" to 0x14, "r" to 0x15, "s" to 0x16, "t" to 0x17,
        "u" to 0x18, "v" to 0x19, "w" to 0x1A, "x" to 0x1B,
        "y" to 0x1C, "z" to 0x1D,
        "1" to 0x1E, "2" to 0x1F, "3" to 0x20, "4" to 0x21, "5" to 0x22,
        "6" to 0x23, "7" to 0x24, "8" to 0x25, "9" to 0x26, "0" to 0x27,
        "enter" to 0x28, "return" to 0x28,
        "esc" to 0x29, "escape" to 0x29,
        "backspace" to 0x2A,
        "tab" to 0x2B,
        "space" to 0x2C,
        "f1" to 0x3A, "f2" to 0x3B, "f3" to 0x3C, "f4" to 0x3D,
        "f5" to 0x3E, "f6" to 0x3F, "f7" to 0x40, "f8" to 0x41,
        "f9" to 0x42, "f10" to 0x43, "f11" to 0x44, "f12" to 0x45,
        "right" to 0x4F, "left" to 0x50, "down" to 0x51, "up" to 0x52,
    )

    /** Modifier-only keys (held in byte 0 of the report). */
    private val modifiers: Map<String, Int> = mapOf(
        "ctrl" to MOD_LCTRL,
        "control" to MOD_LCTRL,
        "left ctrl" to MOD_LCTRL,
        "shift" to MOD_LSHIFT,
        "left shift" to MOD_LSHIFT,
        "alt" to MOD_LALT,
        "left alt" to MOD_LALT,
        "win" to MOD_LGUI,
        "gui" to MOD_LGUI,
        "meta" to MOD_LGUI,
    )

    fun isModifier(name: String): Boolean = modifiers.containsKey(name.lowercase().trim())

    fun modifierBit(name: String): Int = modifiers[name.lowercase().trim()] ?: 0

    fun usageId(name: String): Int = usage[name.lowercase().trim()] ?: 0
}
