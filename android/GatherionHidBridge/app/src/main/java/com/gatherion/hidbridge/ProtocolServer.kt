package com.gatherion.hidbridge

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Line protocol twin of Gatherion SerialHidAdapter / NetworkHidAdapter.
 *
 * SECURITY (F-1601): this socket drives HID input (KEYDOWN/KEYUP/KEY), so it
 * must not accept unauthenticated commands from any LAN client. The bridge
 * is intentionally reachable over Wi-Fi (the PC client connects to the
 * phone's LAN IP, see README / GATHERION_HID_TCP), so loopback-only binding
 * would break the feature; authentication is the fix instead.
 * Pass [authToken]: the first line of every connection must then be
 * `AUTH <token>` or the connection is dropped before any command runs.
 * With no token the server still listens unauthenticated and logs a
 * warning — wiring the token in requires HidBridgeService (caller) and the
 * PC NetworkHidAdapter (client) to agree on it, which is outside this file.
 */
class ProtocolServer(
    private val port: Int = 8765,
    private val onLine: (String) -> String?,
    private val authToken: String? = null,
) {
    companion object {
        private const val TAG = "GatherionHidTcp"
    }

    private val running = AtomicBoolean(false)
    private var server: ServerSocket? = null

    fun start() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "GatherionHidTcp", isDaemon = true) {
            try {
                val ss = ServerSocket(port)
                server = ss
                if (authToken == null) {
                    Log.w(TAG, "listening on $port WITHOUT auth token — any LAN client can send HID input")
                } else {
                    Log.i(TAG, "listening on $port (AUTH required)")
                }
                while (running.get()) {
                    val client = try {
                        ss.accept()
                    } catch (_: Exception) {
                        break
                    }
                    thread(name = "GatherionHidClient", isDaemon = true) {
                        handleClient(client)
                    }
                }
            } catch (exc: Exception) {
                Log.e(TAG, "server failed: $exc")
            } finally {
                running.set(false)
            }
        }
    }

    fun stop() {
        running.set(false)
        try {
            server?.close()
        } catch (_: Exception) {
        }
        server = null
    }

    private fun handleClient(socket: Socket) {
        Log.i(TAG, "client ${socket.inetAddress.hostAddress}")
        try {
            socket.soTimeout = 0
            val reader = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))
            val out = socket.getOutputStream()
            if (authToken != null) {
                val first = reader.readLine()?.trim()
                if (first != "AUTH $authToken") {
                    Log.w(TAG, "client ${socket.inetAddress.hostAddress} rejected: missing/wrong AUTH")
                    return
                }
            }
            while (running.get()) {
                val line = reader.readLine() ?: break
                val trimmed = line.trim()
                if (trimmed.isEmpty()) continue
                val reply = onLine(trimmed)
                if (reply != null) {
                    out.write((reply + "\n").toByteArray(Charsets.UTF_8))
                    out.flush()
                }
            }
        } catch (exc: Exception) {
            Log.d(TAG, "client closed: $exc")
        } finally {
            try {
                socket.close()
            } catch (_: Exception) {
            }
        }
    }
}
