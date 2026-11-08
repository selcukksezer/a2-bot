package com.gatherion.hidbridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothHidDevice
import android.bluetooth.BluetoothHidDeviceAppQosSettings
import android.bluetooth.BluetoothHidDeviceAppSdpSettings
import android.bluetooth.BluetoothProfile
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import java.util.concurrent.Executors

class HidBridgeService : Service() {
    companion object {
        private const val TAG = "GatherionHidSvc"
        private const val CHANNEL_ID = "gatherion_hid"
        private const val NOTIF_ID = 8765
        const val TCP_PORT = 8765
        const val ACTION_STATUS = "com.gatherion.hidbridge.STATUS"

        // Boot keyboard descriptor (8-byte input report id 1)
        private val DESCRIPTOR = byteArrayOf(
            0x05, 0x01, // Usage Page (Generic Desktop)
            0x09, 0x06, // Usage (Keyboard)
            0xA1.toByte(), 0x01, // Collection (Application)
            0x85.toByte(), 0x01, // Report ID (1)
            0x05, 0x07, // Usage Page (Key Codes)
            0x19, 0xE0.toByte(),
            0x29, 0xE7.toByte(),
            0x15, 0x00,
            0x25, 0x01,
            0x75, 0x01,
            0x95.toByte(), 0x08,
            0x81.toByte(), 0x02, // Modifier bits
            0x95.toByte(), 0x01,
            0x75, 0x08,
            0x81.toByte(), 0x01, // Reserved
            0x95.toByte(), 0x06,
            0x75, 0x08,
            0x15, 0x00,
            0x25, 0x65,
            0x05, 0x07,
            0x19, 0x00,
            0x29, 0x65,
            0x81.toByte(), 0x00, // Key array
            0xC0.toByte(),
        )
    }

    inner class LocalBinder : Binder() {
        fun getService(): HidBridgeService = this@HidBridgeService
    }

    private val binder = LocalBinder()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val executor = Executors.newSingleThreadExecutor()

    private var hidProxy: BluetoothHidDevice? = null
    private var hostDevice: BluetoothDevice? = null
    private var keyboard: HidKeyboard? = null
    private var tcp: ProtocolServer? = null
    @Volatile
    var statusText: String = "Starting…"
        private set

    private val profileListener = object : BluetoothProfile.ServiceListener {
        override fun onServiceConnected(profile: Int, proxy: BluetoothProfile?) {
            if (profile != BluetoothProfile.HID_DEVICE || proxy == null) return
            hidProxy = proxy as BluetoothHidDevice
            registerHidApp()
        }

        override fun onServiceDisconnected(profile: Int) {
            if (profile == BluetoothProfile.HID_DEVICE) {
                hidProxy = null
                keyboard = null
                updateStatus("HID profile disconnected")
            }
        }
    }

    private val hidCallback = object : BluetoothHidDevice.Callback() {
        override fun onAppStatusChanged(pluggedDevice: BluetoothDevice?, registered: Boolean) {
            updateStatus(if (registered) "HID app registered — pair/connect PC" else "HID app unregistered")
            if (registered && pluggedDevice != null) {
                connectHost(pluggedDevice)
            }
        }

        override fun onConnectionStateChanged(device: BluetoothDevice?, state: Int) {
            when (state) {
                BluetoothProfile.STATE_CONNECTED -> {
                    hostDevice = device
                    val proxy = hidProxy
                    if (proxy != null && device != null) {
                        keyboard = HidKeyboard(proxy, device)
                    }
                    updateStatus("HID connected: ${device?.name ?: "?"}")
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    if (hostDevice?.address == device?.address) {
                        hostDevice = null
                        keyboard = null
                    }
                    updateStatus("HID disconnected")
                }
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startAsForeground()
        startTcp()
        bindHidProfile()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            "connect" -> connectFirstBonded()
            "test_tab" -> keyboard?.tap("tab")
            "stop" -> {
                stopSelf()
                return START_NOT_STICKY
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        tcp?.stop()
        tcp = null
        try {
            hidProxy?.unregisterApp()
        } catch (_: Exception) {
        }
        val adapter = BluetoothAdapter.getDefaultAdapter()
        try {
            if (hidProxy != null) {
                adapter?.closeProfileProxy(BluetoothProfile.HID_DEVICE, hidProxy)
            }
        } catch (_: Exception) {
        }
        hidProxy = null
        keyboard = null
        super.onDestroy()
    }

    fun connectFirstBonded() {
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: return
        val bonded = try {
            adapter.bondedDevices
        } catch (exc: SecurityException) {
            updateStatus("Bluetooth permission missing")
            return
        }
        val pc = bonded?.firstOrNull()
        if (pc == null) {
            updateStatus("No paired devices — pair PC in system Bluetooth first")
            return
        }
        connectHost(pc)
    }

    private fun connectHost(device: BluetoothDevice) {
        val proxy = hidProxy ?: run {
            updateStatus("HID proxy not ready")
            return
        }
        try {
            val ok = proxy.connect(device)
            updateStatus(if (ok) "Connecting HID → ${device.name}" else "connect() failed")
        } catch (exc: SecurityException) {
            updateStatus("connect denied: $exc")
        }
    }

    private fun bindHidProfile() {
        val adapter = BluetoothAdapter.getDefaultAdapter()
        if (adapter == null || !adapter.isEnabled) {
            updateStatus("Enable Bluetooth")
            return
        }
        val ok = adapter.getProfileProxy(this, profileListener, BluetoothProfile.HID_DEVICE)
        if (!ok) updateStatus("getProfileProxy(HID_DEVICE) failed")
    }

    private fun registerHidApp() {
        val proxy = hidProxy ?: return
        val sdp = BluetoothHidDeviceAppSdpSettings(
            "Gatherion Keyboard",
            "Gatherion bot HID bridge",
            "Gatherion",
            BluetoothHidDevice.SUBCLASS1_KEYBOARD,
            DESCRIPTOR,
        )
        // delayVariation: AOSP MAX_LATENCY_DEFAULT == 0xFFFFFFFF (-1); not on all stubs
        val qos = BluetoothHidDeviceAppQosSettings(
            BluetoothHidDeviceAppQosSettings.SERVICE_BEST_EFFORT,
            800,
            9,
            0,
            11250,
            -1,
        )
        try {
            val ok = proxy.registerApp(sdp, null, qos, executor, hidCallback)
            updateStatus(if (ok) "Registering HID app…" else "registerApp failed")
        } catch (exc: SecurityException) {
            updateStatus("registerApp denied: $exc")
        }
    }

    private fun startTcp() {
        tcp?.stop()
        tcp = ProtocolServer(TCP_PORT) { line -> handleProtocol(line) }.also { it.start() }
        updateStatus("TCP :$TCP_PORT listening")
    }

    private fun handleProtocol(line: String): String? {
        val parts = line.trim().split(Regex("\\s+"))
        if (parts.isEmpty()) return null
        val cmd = parts[0].uppercase()
        val kb = keyboard
        return when (cmd) {
            "PING" -> "PONG"
            "KEYDOWN" -> {
                val key = parts.getOrNull(1) ?: return "ERR missing key"
                if (kb == null) "ERR no HID host"
                else if (kb.keyDown(key)) "OK" else "ERR"
            }
            "KEYUP" -> {
                val key = parts.getOrNull(1) ?: return "ERR missing key"
                if (kb == null) "ERR no HID host"
                else if (kb.keyUp(key)) "OK" else "ERR"
            }
            "KEY" -> {
                val key = parts.getOrNull(1) ?: return "ERR missing key"
                val hold = parts.getOrNull(2)?.toLongOrNull() ?: 80L
                if (kb == null) "ERR no HID host"
                else if (kb.tap(key, hold)) "OK" else "ERR"
            }
            "RELEASE" -> {
                if (kb == null) "ERR no HID host"
                else if (kb.releaseAll()) "OK" else "ERR"
            }
            else -> "ERR unknown $cmd"
        }.also { Log.d(TAG, "$line → $it") }
    }

    private fun updateStatus(text: String) {
        statusText = text
        Log.i(TAG, text)
        mainHandler.post {
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.notify(NOTIF_ID, buildNotification(text))
            sendBroadcast(Intent(ACTION_STATUS).putExtra("status", text).setPackage(packageName))
        }
    }

    private fun createChannel() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        val ch = NotificationChannel(CHANNEL_ID, "Gatherion HID", NotificationManager.IMPORTANCE_LOW)
        nm.createNotificationChannel(ch)
    }

    private fun startAsForeground() {
        val notif = buildNotification(statusText)
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
        } else {
            startForeground(NOTIF_ID, notif)
        }
    }

    private fun buildNotification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notif_title))
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher_vector)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }
}
