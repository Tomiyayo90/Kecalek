package sk.preklad

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.text.InputType
import android.util.Base64
import android.media.AudioAttributes
import android.media.AudioTrack
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.net.URL
import java.util.Locale
import java.util.zip.ZipInputStream
import kotlin.concurrent.thread
import kotlin.math.sqrt

data class Lang(val name: String, val ml: String, val locale: Locale)

class Spk(val male: Boolean, val idx: Int, var hz: Float)

class MainActivity : Activity() {

    companion object {
        const val RATE = 16000
        const val MODEL_NAME = "vosk-model-small-en-us-0.15"
        const val MODEL_URL = "https://alphacephei.com/vosk/models/$MODEL_NAME.zip"
        const val MALE_BELOW_HZ = 165f
        const val GEMINI_MODEL = "gemini-3.1-flash-tts-preview"   // ak Google premenuje model, zmeň tu
        val VOICES_FEMALE = listOf("Kore", "Aoede", "Leda", "Sulafat", "Gacrux")
        val VOICES_MALE = listOf("Charon", "Puck", "Fenrir", "Orus", "Algenib")
        const val SAME_SPEAKER_RATIO = 0.15f   // o koľko % sa smie líšiť výška hlasu, aby to bol ten istý človek
    }

    private val langs = listOf(
        Lang("Čeština", TranslateLanguage.CZECH, Locale("cs", "CZ")),
        Lang("Maďarčina", TranslateLanguage.HUNGARIAN, Locale("hu", "HU")),
        Lang("Slovenčina", TranslateLanguage.SLOVAK, Locale("sk", "SK"))
    )

    private lateinit var csText: TextView
    private lateinit var enText: TextView
    private lateinit var button: Button
    private lateinit var voiceSwitch: Switch
    private lateinit var spinner: Spinner
    private lateinit var keyField: EditText
    private val ttsExecutor = Executors.newSingleThreadExecutor()
    private val pending = AtomicInteger(0)
    @Volatile private var gen = 0

    private var model: Model? = null
    private var modelReady = false
    private var translator: Translator? = null
    private var translatorReady = false
    private var targetLocale = langs[0].locale

    private var tts: TextToSpeech? = null
    private var ttsOk = false
    private var voiceFemale: Voice? = null
    private var voiceMale: Voice? = null

    @Volatile private var running = false
    private val speakers = ArrayList<Spk>()
    private var lastSpk: Spk? = null
    private var lastPartial = ""
    private val history = ArrayDeque<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.BLACK)
            setPadding(32, 48, 32, 32)
        }
        spinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                langs.map { "Preklad do: " + it.name })
        }
        csText = TextView(this).apply {
            textSize = 28f; setTextColor(Color.WHITE); gravity = Gravity.BOTTOM
            text = "Sťahujem modely…"
        }
        enText = TextView(this).apply {
            textSize = 14f; setTextColor(Color.GRAY); setPadding(0, 16, 0, 16)
        }
        voiceSwitch = Switch(this).apply {
            text = "Čítať preklad nahlas (použi slúchadlá!)"
            setTextColor(Color.WHITE); isChecked = true
        }
        keyField = EditText(this).apply {
            hint = "Gemini API kľúč (voliteľné, zadarmo)"
            setHintTextColor(Color.GRAY); setTextColor(Color.WHITE)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setText(getSharedPreferences("p", MODE_PRIVATE).getString("key", ""))
        }
        button = Button(this).apply {
            text = "Štart"; isEnabled = false
            setOnClickListener { if (running) stop() else start() }
        }
        root.addView(spinner, LinearLayout.LayoutParams(-1, -2))
        root.addView(csText, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(enText, LinearLayout.LayoutParams(-1, -2))
        root.addView(voiceSwitch, LinearLayout.LayoutParams(-1, -2))
        root.addView(keyField, LinearLayout.LayoutParams(-1, -2))
        root.addView(button, LinearLayout.LayoutParams(-1, -2))
        setContentView(root)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) setupVoices()
        }
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) = setTarget(pos)
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        loadModel()

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
    }

    // ---------- modely ----------
    private fun loadModel() {
        thread {
            try {
                val dir = File(filesDir, MODEL_NAME)
                if (!File(dir, "am/final.mdl").exists()) {
                    runOnUiThread { csText.text = "Sťahujem rozpoznávací model (~40 MB), len prvýkrát…" }
                    val base = filesDir.canonicalPath
                    URL(MODEL_URL).openStream().use { input ->
                        ZipInputStream(input).use { zip ->
                            var e = zip.nextEntry
                            while (e != null) {
                                val f = File(filesDir, e.name)
                                if (!f.canonicalPath.startsWith(base)) throw Exception("zlý zip")
                                if (e.isDirectory) f.mkdirs() else {
                                    f.parentFile?.mkdirs()
                                    f.outputStream().use { zip.copyTo(it) }
                                }
                                e = zip.nextEntry
                            }
                        }
                    }
                }
                model = Model(dir.absolutePath)
                runOnUiThread { modelReady = true; updateReady() }
            } catch (ex: Exception) {
                runOnUiThread { csText.text = "Chyba modelu: ${ex.message}" }
            }
        }
    }

    private fun setTarget(i: Int) {
        val l = langs[i]
        targetLocale = l.locale
        translatorReady = false
        button.isEnabled = false
        translator?.close()
        val t = Translation.getClient(
            TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(l.ml).build()
        )
        translator = t
        csText.text = "Sťahujem model prekladu (${l.name})…"
        t.downloadModelIfNeeded(DownloadConditions.Builder().build())
            .addOnSuccessListener { translatorReady = true; updateReady() }
            .addOnFailureListener { csText.text = "Chyba prekladu: ${it.message}" }
        setupVoices()
    }

    private fun updateReady() {
        if (modelReady && translatorReady) {
            button.isEnabled = true
            csText.text = "Pripravené. Polož telefón k TV, nasaď slúchadlá a stlač Štart."
        }
    }

    // ---------- hlasy ----------
    private fun setupVoices() {
        val t = tts ?: return
        try {
            val r = t.setLanguage(targetLocale)
            ttsOk = r >= 0
            val vs = t.voices.filter { it.locale.language == targetLocale.language }
                .sortedByDescending { it.quality }
            voiceFemale = vs.getOrNull(0)
            voiceMale = vs.getOrNull(1)
            if (!ttsOk) enText.text = "Chýba hlas pre ${targetLocale.displayLanguage}: Nastavenia > Prevod textu na reč > Google > nainštaluj jazyk."
        } catch (e: Exception) { ttsOk = false }
    }

    private fun systemSpeak(text: String, spk: Spk) {
        val male = spk.male
        val t = tts ?: return
        val twoVoices = voiceMale != null
        val v = if (male) (voiceMale ?: voiceFemale) else voiceFemale
        if (v != null) t.voice = v
        // Ak telefón má len jeden hlas, rozlíš mužský/ženský výškou tónu
        t.setPitch(if (twoVoices) 0.85f + 0.07f * spk.idx else if (male) 0.6f + 0.08f * spk.idx else 1.05f + 0.1f * spk.idx)
        t.speak(text, TextToSpeech.QUEUE_ADD, null, "u" + System.nanoTime())
    }

    // ---------- hlas cez Gemini (zadarmo cez Google AI Studio kľúč) ----------
    private fun speak(text: String, spk: Spk) {
        val key = keyField.text.toString().trim()
        if (key.isEmpty()) { systemSpeak(text, spk); return }
        if (pending.get() >= 3) return              // veľa vo fronte: radšej preskoč, aby hlas nezaostával
        pending.incrementAndGet()
        val g = gen
        ttsExecutor.execute {
            try {
                if (g == gen) {
                    val pcm = geminiTts(key, text, spk)
                    if (g == gen) playPcm(pcm)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    enText.text = "Gemini hlas zlyhal (${e.message}), použil som systémový."
                    if (g == gen) systemSpeak(text, spk)
                }
            } finally { pending.decrementAndGet() }
        }
    }

    private fun geminiTts(key: String, text: String, spk: Spk): ByteArray {
        val voice = (if (spk.male) VOICES_MALE else VOICES_FEMALE)[spk.idx % 5]
        val body = JSONObject()
            .put("contents", JSONArray().put(JSONObject().put("parts",
                JSONArray().put(JSONObject().put("text", "Say naturally and expressively: $text")))))
            .put("generationConfig", JSONObject()
                .put("responseModalities", JSONArray().put("AUDIO"))
                .put("speechConfig", JSONObject().put("voiceConfig", JSONObject()
                    .put("prebuiltVoiceConfig", JSONObject().put("voiceName", voice)))))
        val c = URL("https://generativelanguage.googleapis.com/v1beta/models/$GEMINI_MODEL:generateContent")
            .openConnection() as HttpURLConnection
        c.requestMethod = "POST"
        c.setRequestProperty("Content-Type", "application/json")
        c.setRequestProperty("x-goog-api-key", key)
        c.doOutput = true; c.connectTimeout = 5000; c.readTimeout = 20000
        c.outputStream.use { it.write(body.toString().toByteArray()) }
        if (c.responseCode != 200) throw Exception("HTTP ${c.responseCode}")
        val resp = JSONObject(c.inputStream.bufferedReader().readText())
        val b64 = resp.getJSONArray("candidates").getJSONObject(0).getJSONObject("content")
            .getJSONArray("parts").getJSONObject(0).getJSONObject("inlineData").getString("data")
        return Base64.decode(b64, Base64.DEFAULT)      // PCM 16-bit, 24 kHz, mono
    }

    private fun playPcm(pcm: ByteArray) {
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(24000)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(pcm.size)
            .setTransferMode(AudioTrack.MODE_STATIC).build()
        track.write(pcm, 0, pcm.size)
        track.play()
        Thread.sleep(pcm.size / 2 * 1000L / 24000 + 150)
        track.release()
    }

    // ---------- nahrávanie + rozpoznanie + výška hlasu ----------
    private fun start() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1); return
        }
        val m = model ?: return
        getSharedPreferences("p", MODE_PRIVATE).edit().putString("key", keyField.text.toString().trim()).apply()
        speakers.clear(); lastSpk = null
        running = true
        button.text = "Stop"
        thread {
            val minBuf = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val rec = AudioRecord(MediaRecorder.AudioSource.MIC, RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, 16000))
            val recognizer = Recognizer(m, RATE.toFloat())
            val buf = ShortArray(1600)
            val pitches = ArrayList<Float>()
            rec.startRecording()
            while (running) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) continue
                estimatePitch(buf, n)?.let { pitches.add(it) }
                if (recognizer.acceptWaveForm(buf, n)) {
                    val txt = JSONObject(recognizer.result).optString("text")
                    val hz = medianPitch(pitches); pitches.clear()
                    if (txt.isNotBlank()) runOnUiThread { onSpeech(txt, true, hz) }
                } else {
                    val txt = JSONObject(recognizer.partialResult).optString("partial")
                    if (txt.isNotBlank()) runOnUiThread { onSpeech(txt, false, null) }
                }
            }
            rec.stop(); rec.release(); recognizer.close()
        }
    }

    private fun stop() {
        running = false
        gen++
        button.text = "Štart"
        tts?.stop()
    }

    private fun estimatePitch(b: ShortArray, n: Int): Float? {
        var e = 0.0
        for (i in 0 until n) e += b[i].toDouble() * b[i]
        if (sqrt(e / n) < 600) return null            // ticho / príliš slabý zvuk
        val minLag = RATE / 300
        val maxLag = RATE / 70
        var best = 0.0
        var bestLag = 0
        for (lag in minLag..maxLag) {
            var s = 0.0
            for (i in 0 until n - lag) s += b[i].toDouble() * b[i + lag]
            if (s > best) { best = s; bestLag = lag }
        }
        return if (bestLag > 0 && best / e > 0.35) RATE.toFloat() / bestLag else null
    }

    private fun medianPitch(p: List<Float>): Float? {
        if (p.size < 3) return null
        val sorted = p.sorted()
        return sorted[sorted.size / 2]
    }

    // Priradí vetu ku známemu hovoriacemu (podľa výšky hlasu) alebo vytvorí nového (max 5 mužov + 5 žien)
    private fun assign(hz: Float?): Spk {
        if (hz == null) return lastSpk ?: newSpk(false, 190f)
        val male = hz < MALE_BELOW_HZ
        val same = speakers.filter { it.male == male }
        val near = same.minByOrNull { Math.abs(it.hz - hz) }
        val spk = if (near != null && (Math.abs(near.hz - hz) / near.hz < SAME_SPEAKER_RATIO || same.size >= 5))
            near else newSpk(male, hz)
        spk.hz = spk.hz * 0.7f + hz * 0.3f
        lastSpk = spk
        return spk
    }

    private fun newSpk(male: Boolean, hz: Float): Spk {
        val s = Spk(male, speakers.count { it.male == male }, hz)
        speakers.add(s)
        return s
    }

    // ---------- preklad a výstup ----------
    private fun onSpeech(en: String, final: Boolean, hz: Float?) {
        val spk = if (final) assign(hz) else (lastSpk ?: Spk(false, 0, 190f))
        if (!final) { if (en == lastPartial) return; lastPartial = en }
        val tr = translator
        if (!translatorReady || tr == null) return
        tr.translate(en).addOnSuccessListener { cs ->
            val label = (if (spk.male) "♂" else "♀") + (spk.idx + 1) + " "
            enText.text = label + en
            if (final) {
                history.addLast(label + cs)
                while (history.size > 3) history.removeFirst()
                csText.text = history.joinToString("\n\n")
                if (ttsOk && voiceSwitch.isChecked) speak(cs, spk)
            } else {
                csText.text = (history + (label + cs)).takeLast(3).joinToString("\n\n")
            }
        }
    }

    override fun onDestroy() {
        stop(); tts?.shutdown(); translator?.close(); model?.close()
        super.onDestroy()
    }
}
