package com.gatherion.hidbridge

import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothHidDevice
import android.util.Log
import java.util.concurrent.atomic.AtomicInteger

/**
 * Maintains an 8-byte boot-keyboard report and pushes it via BluetoothHidDevice.
 */
class HidKeyboard(
    private val hidDevice: BluetoothHidDevice,
    private val host: BluetoothDevice?,
) {
    companion object {
        private const val TAG = "GatherionHidKb"
        const val REPORT_ID = 1
    }

    private val modifier = AtomicInteger(0)
    private val keys = IntArray(6)

    @Synchronized
    fun keyDown(name: String): Boolean {
        val key = name.lowercase().trim()
        if (HidKeyMap.isModifier(key)) {
            modifier.set(modifier.get() or HidKeyMap.modifierBit(key))
            return send()
        }
        val usage = HidKeyMap.usageId(key)
        if (usage == 0) {
            Log.w(TAG, "unknown key: $key")
            return false
        }
        if (keys.any { it == usage }) return send()
        val slot = keys.indexOfFirst { it == 0 }
        if (slot < 0) {
            Log.w(TAG, "key rollover full, drop $key")
            return false
        }
        keys[slot] = usage
        return send()
    }

    @Synchronized
    fun keyUp(name: String): Boolean {
        val key = name.lowercase().trim()
        if (HidKeyMap.isModifier(key)) {
            modifier.set(modifier.get() and HidKeyMap.modifierBit(key).inv())
            return send()
        }
        val usage = HidKeyMap.usageId(key)
        if (usage == 0) return false
        for (i in keys.indices) {
            if (keys[i] == usage) keys[i] = 0
        }
        return send()
    }

    @Synchronized
    fun tap(name: String, holdMs: Long = 80L): Boolean {
        if (!keyDown(name)) return false
        try {
            Thread.sleep(holdMs.coerceIn(20L, 500L))
        } catch (_: InterruptedException) {
        }
        return keyUp(name)
    }

    @Synchronized
    fun releaseAll(): Boolean {
        modifier.set(0)
        for (i in keys.indices) keys[i] = 0
        return send()
    }

    private fun send(): Boolean {
        val device = host ?: run {
            Log.w(TAG, "no HID host connected")
            return false
        }
        val report = ByteArray(8)
        report[0] = modifier.get().toByte()
        report[1] = 0
        for (i in 0 until 6) {
            report[2 + i] = keys[i].toByte()
        }
        return try {
            hidDevice.sendReport(device, REPORT_ID, report)
        } catch (exc: SecurityException) {
            Log.e(TAG, "sendReport denied: $exc")
            false
        } catch (exc: Exception) {
            Log.e(TAG, "sendReport failed: $exc")
            false
        }
    }
}
