#include <iostream>
#include <vector>
#include <atomic>
#include <cmath>
#include <thread>
#include <mutex>
#include <memory>
#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <array>

// AAudio headers for the Android backend stub
#if __ANDROID__
#include <aaudio/AAudio.h>
#endif

// ===================================================================================
// MARK: - Constants and Enums
// ===================================================================================

const float PI = 3.14159265358979323846f;
const int MAX_LOOP_SECONDS = 60;
const int MAX_COMMANDS_IN_QUEUE = 32;

enum class LooperState {
    Ready, Recording, Playing, Overdub, Stopped, Waiting, ArmOverdub
};

enum class DelaySyncMode { Note, Milliseconds };

enum class DelayVariant { Straight, Dotted, Triplet };

struct StereoSample { float left = 0.f; float right = 0.f; };

// ===================================================================================
// MARK: - Utility & DSP Primitives (Real-Time Safe)
// ===================================================================================

namespace DSPUtils {
    // Clamps a value between a low and high bound.
    inline float clamp(float value, float low, float high) {
        return std::max(low, std::min(value, high));
    }

    // A one-pole filter for smoothing parameter changes to prevent clicks.
    template <typename T>
    class SmoothedValue {
        T current = T(0), target = T(0);
        float smoothness = 0.99f; // default smoothness
    public:
        T getNext() {
            current = (current * smoothness) + (target * (1.f - smoothness));
            return current;
        }
        T getTargetValue() const { return target; }
        void setTarget(T newTarget) { target = newTarget; }
        void setSmoothness(float sampleRate, float timeMs) {
            smoothness = expf(-1000.f / (sampleRate * timeMs));
        }
        void setCurrentAndTarget(T val) { current = target = val; }
    };

    // A simple, fixed-size circular buffer for audio delay lines.
    template <typename T>
    class CircularBuffer {
        std::vector<T> buffer;
        size_t writeIndex = 0;
        size_t mask;
    public:
        CircularBuffer(size_t size = 0) {
            if (size > 0) resize(size);
        }
        void resize(size_t newSize) {
            size_t powerOfTwoSize = 1;
            while (powerOfTwoSize < newSize) powerOfTwoSize <<= 1;
            buffer.assign(powerOfTwoSize, T{});
            mask = powerOfTwoSize - 1;
            writeIndex = 0;
        }
        void write(T sample) {
            buffer[writeIndex] = sample;
            writeIndex = (writeIndex + 1) & mask;
        }
        T read(int delaySamples) const {
            if (buffer.empty()) return T{};
            size_t readIndex = (writeIndex - delaySamples + buffer.size()) & mask;
            return buffer[readIndex];
        }
        size_t size() const { return buffer.size(); }
    };

    // Low-Frequency Oscillator (LFO).
    class LFO {
        float phase = 0.0f;
        float sampleRate = 48000.0f;
    public:
        void setSampleRate(float sr) { sampleRate = sr; }
        float process(float frequency) {
            phase += frequency / sampleRate;
            if (phase >= 1.0f) phase -= 1.0f;
            return sinf(2.0f * PI * phase);
        }
    };

    // Generic Biquad Filter.
    class BiquadFilter {
        float a0 = 1.f, a1 = 0.f, a2 = 0.f, b1 = 0.f, b2 = 0.f;
        float z1L = 0.f, z2L = 0.f, z1R = 0.f, z2R = 0.f; // state
    public:
        void setCoefficients(float new_a0, float new_a1, float new_a2, float new_b1, float new_b2) {
            a0 = new_a0; a1 = new_a1; a2 = new_a2; b1 = new_b1; b2 = new_b2;
        }
        void setLowShelf(float sr, float freq, float gainDB, float q = 0.707f) {
            float A = powf(10.0f, gainDB / 40.0f);
            float w0 = 2.0f * PI * freq / sr;
            float cos_w0 = cosf(w0);
            float sin_w0 = sinf(w0);
            float alpha = sin_w0 / (2.0f * q);
            float beta = 2.0f * sqrtf(A) * alpha;
            float b0_ = A * ((A + 1.f) - (A - 1.f) * cos_w0 + beta);
            float b1_ = 2.f * A * ((A - 1.f) - (A + 1.f) * cos_w0);
            float b2_ = A * ((A + 1.f) - (A - 1.f) * cos_w0 - beta);
            float a0_ = (A + 1.f) + (A - 1.f) * cos_w0 + beta;
            float a1_ = -2.f * ((A - 1.f) + (A + 1.f) * cos_w0);
            float a2_ = (A + 1.f) + (A - 1.f) * cos_w0 - beta;
            setCoefficients(b0_ / a0_, b1_ / a0_, b2_ / a0_, a1_ / a0_, a2_ / a0_);
        }
        void setPeaking(float sr, float freq, float gainDB, float q) {
            float A = powf(10.0f, gainDB / 40.0f);
            float w0 = 2.0f * PI * freq / sr;
            float cos_w0 = cosf(w0);
            float sin_w0 = sinf(w0);
            float alpha = sin_w0 / (2.0f * q);
            float b0_ = 1.f + alpha * A;
            float b1_ = -2.f * cos_w0;
            float b2_ = 1.f - alpha * A;
            float a0_ = 1.f + alpha / A;
            float a1_ = -2.f * cos_w0;
            float a2_ = 1.f - alpha / A;
            setCoefficients(b0_ / a0_, b1_ / a0_, b2_ / a0_, a1_ / a0_, a2_ / a0_);
        }
        void setHighShelf(float sr, float freq, float gainDB, float q = 0.707f) {
            float A = powf(10.0f, gainDB / 40.0f);
            float w0 = 2.0f * PI * freq / sr;
            float cos_w0 = cosf(w0);
            float sin_w0 = sinf(w0);
            float alpha = sin_w0 / (2.0f * q);
            float beta = 2.0f * sqrtf(A) * alpha;
            float b0_ = A * ((A + 1.f) + (A - 1.f) * cos_w0 + beta);
            float b1_ = -2.f * A * ((A - 1.f) + (A + 1.f) * cos_w0);
            float b2_ = A * ((A + 1.f) + (A - 1.f) * cos_w0 - beta);
            float a0_ = (A + 1.f) - (A - 1.f) * cos_w0 + beta;
            float a1_ = 2.f * ((A - 1.f) + (A + 1.f) * cos_w0);
            float a2_ = (A + 1.f) - (A - 1.f) * cos_w0 - beta;
            setCoefficients(b0_ / a0_, b1_ / a0_, b2_ / a0_, a1_ / a0_, a2_ / a0_);
        }
        void setLowPass(float sr, float freq, float q) {
            float w0 = 2.0f * PI * freq / sr;
            float cos_w0 = cosf(w0);
            float alpha = sinf(w0) / (2.0f * q);
            float b1_ = 1.f - cos_w0;
            float b0_ = b1_ / 2.f;
            float b2_ = b0_;
            float a0_ = 1.f + alpha;
            float a1_ = -2.f * cos_w0;
            float a2_ = 1.f - alpha;
            setCoefficients(b0_ / a0_, b1_ / a0_, b2_ / a0_, a1_ / a0_, a2_ / a0_);
        }
        void setHighPass(float sr, float freq, float q) {
            float w0 = 2.0f * PI * freq / sr;
            float cos_w0 = cosf(w0);
            float alpha = sinf(w0) / (2.0f * q);
            float b0_ = (1.f + cos_w0) / 2.f;
            float b1_ = -(1.f + cos_w0);
            float b2_ = b0_;
            float a0_ = 1.f + alpha;
            float a1_ = -2.f * cos_w0;
            float a2_ = 1.f - alpha;
            setCoefficients(b0_ / a0_, b1_ / a0_, b2_ / a0_, a1_ / a0_, a2_ / a0_);
        }

        StereoSample process(StereoSample x) {
            float inL = x.left;
            float inR = x.right;
            float outL = a0 * inL + a1 * z1L + a2 * z2L - b1 * z1L - b2 * z2L;
            z2L = z1L;
            z1L = inL;
            float outR = a0 * inR + a1 * z1R + a2 * z2R - b1 * z1R - b2 * z2R;
            z2R = z1R;
            z1R = inR;
            return {outL, outR};
        }
    };
}

// ===================================================================================
// MARK: - Command Queue for Lock-Free UI->Audio Communication
// ===================================================================================
enum class CommandType { HandleMain, HandleStop, Clear, SetVolume, AddFx, RemoveFx, MoveFx, ToggleBypassFx, SetFxParam };

struct Command {
    CommandType type;
    int trackIdx;
    int fxId = -1;
    int paramIndex = -1;
    float value = 0.f;
    std::string stringValue;
};

class SPSCCommandQueue {
    std::array<Command, MAX_COMMANDS_IN_QUEUE> commands;
    std::atomic<int> readPos{0}, writePos{0};
public:
    bool push(const Command& cmd) {
        int nextWritePos = (writePos.load(std::memory_order_relaxed) + 1) % MAX_COMMANDS_IN_QUEUE;
        if (nextWritePos == readPos.load(std::memory_order_acquire)) return false; // Queue full
        commands[writePos.load(std::memory_order_relaxed)] = cmd;
        writePos.store(nextWritePos, std::memory_order_release);
        return true;
    }
    bool pop(Command& cmd) {
        if (readPos.load(std::memory_order_acquire) == writePos.load(std::memory_order_acquire)) return false; // Queue empty
        cmd = commands[readPos.load(std::memory_order_relaxed)];
        readPos.store((readPos.load(std::memory_order_relaxed) + 1) % MAX_COMMANDS_IN_QUEUE, std::memory_order_release);
        return true;
    }
};

// ===================================================================================
// MARK: - Abstract Effect Base Class & All Implementations
// ===================================================================================

class Effect {
public:
    int id;
    std::string type;
    std::string name;
    std::atomic<bool> bypass{false};

    Effect(int an_id, std::string a_type, std::string a_name)
        : id(an_id), type(std::move(a_type)), name(std::move(a_name)) {}
    virtual ~Effect() = default;
    virtual StereoSample process(StereoSample sample) = 0;
    virtual void setSampleRate(float sr) {}
    virtual void setParam(int paramIndex, float value) {}
    virtual std::unique_ptr<Effect> clone() const = 0;
};

// --- FDN Reverb ---
class Reverb : public Effect {
    static const int NUM_DELAYS = 4;
    std::array<DSPUtils::CircularBuffer<float>, NUM_DELAYS> delaysL, delaysR;
    std::array<float, NUM_DELAYS> feedbackGains;
    DSPUtils::SmoothedValue<float> mix;
    float sampleRate = 48000.f;

public:
    Reverb(int id) : Effect(id, "Reverb", "Reverb") { mix.setCurrentAndTarget(0.25f); }

    void setSampleRate(float sr) override {
        sampleRate = sr;
        std::array<int, NUM_DELAYS> delaySamples = { (int)(sr * 0.0297f), (int)(sr * 0.0371f), (int)(sr * 0.0411f), (int)(sr * 0.0437f) };
        for(int i=0; i<NUM_DELAYS; ++i) {
            delaysL[i].resize(delaySamples[i]);
            delaysR[i].resize(delaySamples[i]);
            feedbackGains[i] = powf(10.f, -3.f * delaySamples[i] / (sr * 2.5f));
        }
        mix.setSmoothness(sr, 20.f);
    }

    void setParam(int paramIndex, float value) override {
        if (paramIndex == 0) mix.setTarget(value);
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        float m = mix.getNext();
        if (m < 0.001f) return sample;

        float inputL = sample.left;
        float inputR = sample.right;

        std::array<float, NUM_DELAYS> outsL, outsR;
        for(int i=0; i<NUM_DELAYS; ++i) {
            outsL[i] = delaysL[i].read(0);
            outsR[i] = delaysR[i].read(0);
        }

        // Simple Hadamard matrix mixing
        float m1_L = outsL[0] + outsL[1];
        float m2_L = outsL[0] - outsL[1];
        float m3_L = outsL[2] + outsL[3];
        float m4_L = outsL[2] - outsL[3];
        delaysL[0].write(inputL + (m1_L + m3_L) * 0.5f * feedbackGains[0]);
        delaysL[1].write(inputL + (m2_L + m4_L) * 0.5f * feedbackGains[1]);
        delaysL[2].write(inputL + (m1_L - m3_L) * 0.5f * feedbackGains[2]);
        delaysL[3].write(inputL + (m2_L - m4_L) * 0.5f * feedbackGains[3]);

        float m1_R = outsR[0] + outsR[1];
        float m2_R = outsR[0] - outsR[1];
        float m3_R = outsR[2] + outsR[3];
        float m4_R = outsR[2] - outsR[3];
        delaysR[0].write(inputR + (m1_R + m3_R) * 0.5f * feedbackGains[0]);
        delaysR[1].write(inputR + (m2_R + m4_R) * 0.5f * feedbackGains[1]);
        delaysR[2].write(inputR + (m1_R - m3_R) * 0.5f * feedbackGains[2]);
        delaysR[3].write(inputR + (m2_R - m4_R) * 0.5f * feedbackGains[3]);

        float wetL = (outsL[0] - outsL[1] + outsL[2] - outsL[3]) * 0.5f;
        float wetR = (outsR[0] - outsR[1] + outsR[2] - outsR[3]) * 0.5f;

        sample.left = sample.left * (1.f - m) + wetL * m;
        sample.right = sample.right * (1.f - m) + wetR * m;

        return sample;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<Reverb>(id);
        newFx->setSampleRate(sampleRate);
        newFx->mix.setCurrentAndTarget(mix.getTargetValue());
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- Tempo-Synced Delay ---
class Delay : public Effect {
    DSPUtils::CircularBuffer<StereoSample> buffer;
    float sampleRate = 48000.0f;
    DSPUtils::SmoothedValue<float> mix;
    DSPUtils::SmoothedValue<float> feedback;
    DSPUtils::SmoothedValue<float> delayTimeSec;

    std::atomic<DelaySyncMode> syncMode{DelaySyncMode::Milliseconds};
    std::atomic<DelayVariant> variant{DelayVariant::Straight};
    std::string division = "1/8";
    const std::map<std::string, float> NOTE_MULT = {
        {"1/1", 4.0f}, {"1/2", 2.0f}, {"1/4", 1.0f},
        {"1/8", 0.5f}, {"1/16", 0.25f}, {"1/32", 0.125f}
    };
    float applyVariant(float mult, DelayVariant v) {
        if (v == DelayVariant::Dotted) return mult * 1.5f;
        if (v == DelayVariant::Triplet) return mult * (2.0f / 3.0f);
        return mult;
    }

public:
    Delay(int id, std::string name = "Delay") : Effect(id, "Delay", name) {
        mix.setCurrentAndTarget(0.25f);
        feedback.setCurrentAndTarget(0.35f);
        delayTimeSec.setCurrentAndTarget(0.25f);
    }

    void setSampleRate(float sr) override {
        sampleRate = sr;
        buffer.resize((int)(sr * 2.0f));
        mix.setSmoothness(sr, 20.f);
        feedback.setSmoothness(sr, 20.f);
        delayTimeSec.setSmoothness(sr, 20.f);
    }

    void updateFromBPM(float bpm) {
        if (syncMode.load() != DelaySyncMode::Note || bpm <= 0.0f) { return; }
        float quarterSec = 60.0f / bpm;
        float mult = NOTE_MULT.count(division)? NOTE_MULT.at(division) : 0.5f;
        float finalMult = applyVariant(mult, variant.load());
        delayTimeSec.setTarget(DSPUtils::clamp(quarterSec * finalMult, 0.001f, 2.0f));
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        int delaySamples = static_cast<int>(delayTimeSec.getNext() * sampleRate);

        StereoSample delayed = buffer.read(delaySamples);
        float currentFeedback = feedback.getNext();
        StereoSample newSample = {
            sample.left + delayed.left * currentFeedback,
            sample.right + delayed.right * currentFeedback
        };
        buffer.write(newSample);

        float currentMix = mix.getNext();
        sample.left = sample.left * (1.f - currentMix) + delayed.left * currentMix;
        sample.right = sample.right * (1.f - currentMix) + delayed.right * currentMix;
        return sample;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<Delay>(id, name);
        newFx->setSampleRate(sampleRate);
        newFx->mix.setCurrentAndTarget(mix.getTargetValue());
        newFx->feedback.setCurrentAndTarget(feedback.getTargetValue());
        newFx->delayTimeSec.setCurrentAndTarget(delayTimeSec.getTargetValue());
        newFx->syncMode = syncMode.load();
        newFx->division = division;
        newFx->variant = variant.load();
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- Flanger ---
class Flanger : public Effect {
    DSPUtils::CircularBuffer<StereoSample> buffer;
    DSPUtils::LFO lfo;
    DSPUtils::SmoothedValue<float> mix, rateHz, depthMs, feedback;
    float sampleRate = 48000.0f;
public:
    Flanger(int id) : Effect(id, "Flanger", "Flanger") {
        mix.setCurrentAndTarget(0.22f); rateHz.setCurrentAndTarget(0.25f);
        depthMs.setCurrentAndTarget(2.0f); feedback.setCurrentAndTarget(0.0f);
    }

    void setSampleRate(float sr) override {
        sampleRate = sr;
        lfo.setSampleRate(sr);
        buffer.resize((int)(sr * 0.02f)); // Max 20ms flanger delay
        mix.setSmoothness(sr, 20.f); rateHz.setSmoothness(sr, 20.f);
        depthMs.setSmoothness(sr, 20.f); feedback.setSmoothness(sr, 20.f);
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        float lfoVal = (lfo.process(rateHz.getNext()) + 1.f) / 2.f; // 0 to 1
        float delayMs = 1.0f + lfoVal * depthMs.getNext(); // Base 1ms delay
        float delaySamples = delayMs / 1000.f * sampleRate;

        StereoSample delayed = buffer.read(static_cast<int>(delaySamples));
        float currentFeedback = feedback.getNext();
        StereoSample newSample = {
            sample.left + delayed.left * currentFeedback,
            sample.right + delayed.right * currentFeedback
        };
        buffer.write(newSample);

        float currentMix = mix.getNext();
        sample.left = sample.left * (1.f - currentMix) + delayed.left * currentMix;
        sample.right = sample.right * (1.f - currentMix) + delayed.right * currentMix;
        return sample;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<Flanger>(id);
        newFx->setSampleRate(sampleRate);
        newFx->mix.setCurrentAndTarget(mix.getTargetValue());
        newFx->rateHz.setCurrentAndTarget(rateHz.getTargetValue());
        newFx->depthMs.setCurrentAndTarget(depthMs.getTargetValue());
        newFx->feedback.setCurrentAndTarget(feedback.getTargetValue());
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- 3-Band EQ ---
class EQ3Band : public Effect {
    DSPUtils::BiquadFilter low, mid, high;
    DSPUtils::SmoothedValue<float> lowGain, midGain, midFreq, midQ, highGain;
    float sampleRate = 48000.0f;
    bool needsUpdate = true;
public:
    EQ3Band(int id) : Effect(id, "EQ", "3-Band EQ") {
        lowGain.setCurrentAndTarget(3.f); midGain.setCurrentAndTarget(2.f); midFreq.setCurrentAndTarget(1200.f); midQ.setCurrentAndTarget(0.9f); highGain.setCurrentAndTarget(3.f);
    }

    void setSampleRate(float sr) override {
        sampleRate = sr;
        lowGain.setSmoothness(sr, 20.f); midGain.setSmoothness(sr, 20.f); midFreq.setSmoothness(sr, 20.f); midQ.setSmoothness(sr, 20.f); highGain.setSmoothness(sr, 20.f);
        updateCoeffs();
    }

    void updateCoeffs() {
        low.setLowShelf(sampleRate, 180.f, lowGain.getNext());
        mid.setPeaking(sampleRate, midFreq.getNext(), midGain.getNext(), midQ.getNext());
        high.setHighShelf(sampleRate, 4500.f, highGain.getNext());
        needsUpdate = false;
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        if (needsUpdate) updateCoeffs();

        sample = low.process(sample);
        sample = mid.process(sample);
        sample = high.process(sample);
        return sample;
    }

    void setParam(int paramIndex, float value) override {
        switch(paramIndex) {
            case 0: lowGain.setTarget(value); break;
            case 1: midGain.setTarget(value); break;
            case 2: midFreq.setTarget(value); break;
            case 3: midQ.setTarget(value); break;
            case 4: highGain.setTarget(value); break;
        }
        needsUpdate = true;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<EQ3Band>(id);
        newFx->setSampleRate(sampleRate);
        newFx->lowGain.setCurrentAndTarget(lowGain.getTargetValue());
        newFx->midGain.setCurrentAndTarget(midGain.getTargetValue());
        newFx->midFreq.setCurrentAndTarget(midFreq.getTargetValue());
        newFx->midQ.setCurrentAndTarget(midQ.getTargetValue());
        newFx->highGain.setCurrentAndTarget(highGain.getTargetValue());
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- Real-Time Pitch Shifter (Granular Overlap-Add) ---
class RealTimePitchShifter : public Effect {
    static const int GRAIN_SIZE = 1024;
    static const int HOP_SIZE = GRAIN_SIZE / 4;

    std::vector<float> window;
    std::vector<float> inBufferL, inBufferR;
    std::vector<float> outBufferL, outBufferR;
    int inWritePos = 0, outReadPos = 0;

    std::atomic<float> pitchShiftRatio{1.0f};

public:
    RealTimePitchShifter(int id) : Effect(id, "Pitch", "Pitch Shifter") {
        window.resize(GRAIN_SIZE);
        for (int i = 0; i < GRAIN_SIZE; ++i) {
            window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (GRAIN_SIZE - 1)));
        }
        inBufferL.assign(GRAIN_SIZE * 2, 0.f); inBufferR.assign(GRAIN_SIZE * 2, 0.f);
        outBufferL.assign(GRAIN_SIZE * 2, 0.f); outBufferR.assign(GRAIN_SIZE * 2, 0.f);
    }

    void setParam(int paramIndex, float value) override {
        if (paramIndex == 0) { // Semitones
            pitchShiftRatio = powf(2.0f, value / 12.0f);
        }
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load() || std::abs(pitchShiftRatio.load() - 1.0f) < 0.001f) {
            return sample;
        }

        inBufferL[inWritePos] = sample.left;
        inBufferR[inWritePos] = sample.right;

        sample.left = outBufferL[outReadPos];
        sample.right = outBufferR[outReadPos];

        inWritePos++;
        outReadPos++;

        if (inWritePos >= GRAIN_SIZE) {
            std::copy(inBufferL.begin() + HOP_SIZE, inBufferL.begin() + GRAIN_SIZE, inBufferL.begin());
            std::copy(inBufferR.begin() + HOP_SIZE, inBufferR.begin() + GRAIN_SIZE, inBufferR.begin());
            inWritePos -= HOP_SIZE;
        }

        if (outReadPos >= HOP_SIZE) {
            std::copy(outBufferL.begin() + HOP_SIZE, outBufferL.end(), outBufferL.begin());
            std::copy(outBufferR.begin() + HOP_SIZE, outBufferR.end(), outBufferR.begin());
            std::fill(outBufferL.end() - HOP_SIZE, outBufferL.end(), 0.f);
            std::fill(outBufferR.end() - HOP_SIZE, outBufferR.end(), 0.f);
            outReadPos -= HOP_SIZE;

            for (int i = 0; i < GRAIN_SIZE; i++) {
                float readPos = (float)i / pitchShiftRatio.load();
                long   index1 = static_cast<long>(readPos);
                long   index2 = index1 + 1;
                float  frac   = readPos - index1;

                if (index2 < GRAIN_SIZE) {
                    float sL = inBufferL[index1] * (1.f - frac) + inBufferL[index2] * frac;
                    float sR = inBufferR[index1] * (1.f - frac) + inBufferR[index2] * frac;
                    outBufferL[i] += sL * window[i];
                    outBufferR[i] += sR * window[i];
                }
            }
        }
        return sample;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<RealTimePitchShifter>(id);
        newFx->pitchShiftRatio = pitchShiftRatio.load();
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- Low/High Pass Filters ---
class Filter : public Effect {
    DSPUtils::BiquadFilter filter;
    bool isLowPass;
    DSPUtils::SmoothedValue<float> cutoff, q;
    float sampleRate = 48000.f;
    bool needsUpdate = true;
public:
    Filter(int id, bool lp) : Effect(id, lp? "LowPass" : "HighPass", lp? "Low-pass Filter" : "High-pass Filter"), isLowPass(lp) {
        if (!lp) cutoff.setCurrentAndTarget(120.f);
        else cutoff.setCurrentAndTarget(12000.f);
        q.setCurrentAndTarget(0.7f);
    }

    void setSampleRate(float sr) override {
        sampleRate = sr;
        cutoff.setSmoothness(sr, 20.f);
        q.setSmoothness(sr, 20.f);
        updateCoeffs();
    }

    void setParam(int paramIndex, float value) override {
        if(paramIndex == 0) cutoff.setTarget(value);
        if(paramIndex == 1) q.setTarget(value);
        needsUpdate = true;
    }

    void updateCoeffs() {
        if (isLowPass) filter.setLowPass(sampleRate, cutoff.getNext(), q.getNext());
        else filter.setHighPass(sampleRate, cutoff.getNext(), q.getNext());
        needsUpdate = false;
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        if (needsUpdate) updateCoeffs();
        return filter.process(sample);
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<Filter>(id, isLowPass);
        newFx->setSampleRate(sampleRate);
        newFx->cutoff.setCurrentAndTarget(cutoff.getTargetValue());
        newFx->q.setCurrentAndTarget(q.getTargetValue());
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- Panner ---
class Panner : public Effect {
    DSPUtils::SmoothedValue<float> pan;
public:
    Panner(int id) : Effect(id, "Pan", "Pan") { pan.setCurrentAndTarget(0.f); }

    void setSampleRate(float sr) override {
        pan.setSmoothness(sr, 20.f);
    }

    void setParam(int paramIndex, float value) override {
        if (paramIndex == 0) pan.setTarget(DSPUtils::clamp(value, -1.f, 1.f));
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        float currentPan = pan.getNext();
        float angle = (currentPan * 0.5f + 0.5f) * PI / 2.0f;
        sample.left *= cosf(angle);
        sample.right *= sinf(angle);
        return sample;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<Panner>(id);
        newFx->setSampleRate(sampleRate);
        newFx->pan.setCurrentAndTarget(pan.getTargetValue());
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// --- Compressor ---
class Compressor : public Effect {
    DSPUtils::SmoothedValue<float> threshold, knee, ratio, attack, release;
    float envelope = 0.f;
    float sampleRate = 48000.f;
public:
    Compressor(int id) : Effect(id, "Compressor", "Compressor") {
        threshold.setCurrentAndTarget(-18.f); knee.setCurrentAndTarget(6.f); ratio.setCurrentAndTarget(3.f);
        attack.setCurrentAndTarget(0.003f); release.setCurrentAndTarget(0.25f);
    }

    void setSampleRate(float sr) override {
        sampleRate = sr;
        threshold.setSmoothness(sr, 20.f); knee.setSmoothness(sr, 20.f); ratio.setSmoothness(sr, 20.f);
        attack.setSmoothness(sr, 20.f); release.setSmoothness(sr, 20.f);
    }

    void setParam(int paramIndex, float value) override {
        switch(paramIndex) {
            case 0: threshold.setTarget(value); break;
            case 1: ratio.setTarget(value); break;
            case 2: attack.setTarget(value); break;
            case 3: release.setTarget(value); break;
        }
    }

    StereoSample process(StereoSample sample) override {
        if (bypass.load()) return sample;
        float inputMax = std::max(std::abs(sample.left), std::abs(sample.right));
        float inputDB = 20.f * log10f(inputMax + 1e-9);
        float gainReduction = 0;

        float currentThreshold = threshold.getNext();
        float currentKnee = knee.getNext();
        float currentRatio = ratio.getNext();

        if (inputDB > currentThreshold - currentKnee / 2.f) {
            if (inputDB < currentThreshold + currentKnee / 2.f) {
                float x = inputDB - (currentThreshold - currentKnee / 2.f);
                gainReduction = (1.f / currentRatio - 1.f) * (x * x) / (2.f * currentKnee);
            } else {
                gainReduction = (1.f / currentRatio - 1.f) * (inputDB - currentThreshold);
            }
        }

        float attackCoeff = expf(-1.f / (attack.getNext() * sampleRate));
        float releaseCoeff = expf(-1.f / (release.getNext() * sampleRate));

        if (gainReduction < envelope) {
            envelope = attackCoeff * envelope + (1 - attackCoeff) * gainReduction;
        } else {
            envelope = releaseCoeff * envelope + (1 - releaseCoeff) * gainReduction;
        }

        float gain = powf(10.f, envelope / 20.f);
        sample.left *= gain;
        sample.right *= gain;
        return sample;
    }

    std::unique_ptr<Effect> clone() const override {
        auto newFx = std::make_unique<Compressor>(id);
        newFx->setSampleRate(sampleRate);
        newFx->threshold.setCurrentAndTarget(threshold.getTargetValue());
        newFx->knee.setCurrentAndTarget(knee.getTargetValue());
        newFx->ratio.setCurrentAndTarget(ratio.getTargetValue());
        newFx->attack.setCurrentAndTarget(attack.getTargetValue());
        newFx->release.setCurrentAndTarget(release.getTargetValue());
        newFx->bypass = bypass.load();
        return newFx;
    }
};

// ===================================================================================
// MARK: - Looper Track Class
// ===================================================================================

class LooperTrack {
public:
    int index;
    std::atomic<LooperState> state{LooperState::Ready};
    std::vector<float> bufferL, bufferR;
    long loopDurationSamples = 0;
    long writeIndex = 0;
    double readIndex = 0;
    bool justWrapped = false;

    using FxChainPtr = std::shared_ptr<std::vector<std::unique_ptr<Effect>>>;
    std::atomic<FxChainPtr> fxChain;
    DSPUtils::SmoothedValue<float> volume;
    int nextFxId = 1;

    LooperTrack(int idx, float sr) : index(idx) {
        bufferL.assign(sr * MAX_LOOP_SECONDS, 0.0f);
        bufferR.assign(sr * MAX_LOOP_SECONDS, 0.0f);
        fxChain.store(std::make_shared<std::vector<std::unique_ptr<Effect>>>());
        volume.setCurrentAndTarget(0.9f);
        volume.setSmoothness(sr, 15.f);
    }

    void clear() {
        state = LooperState::Ready;
        std::fill(bufferL.begin(), bufferL.end(), 0.0f);
        std::fill(bufferR.begin(), bufferR.end(), 0.0f);
        loopDurationSamples = 0;
        writeIndex = 0; readIndex = 0;
        fxChain.store(std::make_shared<std::vector<std::unique_ptr<Effect>>>());
        nextFxId = 1;
    }

    void process(StereoSample& outSample) {
        justWrapped = false;
        if (state.load() != LooperState::Playing && state.load() != LooperState::Overdub) {
            outSample = {0.f, 0.f};
            return;
        }
        if (loopDurationSamples == 0) return;

        long current_idx = static_cast<long>(readIndex);
        outSample = { bufferL[current_idx], bufferR[current_idx] };

        auto currentFxChain = fxChain.load();
        if(currentFxChain) {
            for (auto& fx : *currentFxChain) {
                if(fx) outSample = fx->process(outSample);
            }
        }

        float vol = volume.getNext();
        outSample.left *= vol;
        outSample.right *= vol;

        readIndex += 1.0;
        if (readIndex >= loopDurationSamples) {
            readIndex -= loopDurationSamples;
            justWrapped = true;
        }
    }
};

// ===================================================================================
// MARK: - Main Looper Engine
// ===================================================================================

class LooperEngine {
public:
    static const int NUM_TRACKS = 4;
    std::vector<LooperTrack> tracks;
    SPSCCommandQueue commandQueue;

    std::atomic<long> globalSampleCounter{0};
    std::atomic<long> masterLoopDurationSamples{0};
    std::atomic<long> masterLoopStartSample{0};
    std::atomic<float> masterBPM{120.f};
    float sampleRate = 48000.0f;

    // Global Before-FX
    Reverb beforeFxReverb;
    Delay beforeFxDelay;
    Flanger beforeFxFlanger;
    EQ3Band beforeFxEQ;
    std::atomic<bool> beforeFxReverbOn{false}, beforeFxDelayOn{false}, beforeFxFlangerOn{false}, beforeFxEQOn{false};
    std::atomic<bool> liveMonitoringOn{false};

    LooperEngine(float sr = 48000.0f) :
        beforeFxReverb(0), beforeFxDelay(0, "Before-FX Delay"), beforeFxFlanger(0), beforeFxEQ(0)
    {
        setSampleRate(sr);
        for (int i = 0; i < NUM_TRACKS; i++) {
            tracks.emplace_back(i, sampleRate);
        }
    }

    void setSampleRate(float sr) {
        sampleRate = sr;
        beforeFxReverb.setSampleRate(sr);
        beforeFxDelay.setSampleRate(sr);
        beforeFxFlanger.setSampleRate(sr);
        beforeFxEQ.setSampleRate(sr);
    }

    void process(const float* input, float* output, int numFrames) {
        for (int f = 0; f < numFrames; ++f) {
            if (f == 0) {
                Command cmd;
                while (commandQueue.pop(cmd)) {
                    processCommand(cmd);
                }
            }

            long currentSample = globalSampleCounter++;
            StereoSample inSample = { input ? input[f*2] : 0.f, input ? input[f*2 + 1] : 0.f };

            StereoSample processedSample = inSample;
            if (beforeFxEQOn.load()) processedSample = beforeFxEQ.process(processedSample);
            if (beforeFxDelayOn.load()) processedSample = beforeFxDelay.process(processedSample);
            if (beforeFxFlangerOn.load()) processedSample = beforeFxFlanger.process(processedSample);
            if (beforeFxReverbOn.load()) processedSample = beforeFxReverb.process(processedSample);

            StereoSample recordSignal = processedSample;

            StereoSample mixedOut = {0.f, 0.f};
            for (auto& track : tracks) {
                LooperState currentState = track.state.load();
                if (currentState == LooperState::Waiting && masterLoopDurationSamples.load() > 0) {
                    if ((currentSample - masterLoopStartSample.load()) % masterLoopDurationSamples.load() == 0) {
                        track.state = LooperState::Recording;
                        track.writeIndex = 0;
                    }
                } else if (currentState == LooperState::ArmOverdub && track.justWrapped) {
                    track.state = LooperState::Overdub;
                    track.writeIndex = static_cast<long>(track.readIndex);
                }

                if ((currentState == LooperState::Recording || currentState == LooperState::Overdub) && track.writeIndex < track.bufferL.size()) {
                    if (currentState == LooperState::Recording) {
                        track.bufferL[track.writeIndex] = 0;
                        track.bufferR[track.writeIndex] = 0;
                    }
                    track.bufferL[track.writeIndex] += recordSignal.left;
                    track.bufferR[track.writeIndex] += recordSignal.right;
                    track.writeIndex++;
                }

                StereoSample trackOut;
                track.process(trackOut);
                mixedOut.left += trackOut.left;
                mixedOut.right += trackOut.right;
            }

            output[f*2] = mixedOut.left + (liveMonitoringOn.load() ? recordSignal.left : 0.f);
            output[f*2 + 1] = mixedOut.right + (liveMonitoringOn.load() ? recordSignal.right : 0.f);
        }
    }

private:
    void processCommand(const Command& cmd) {
        if (cmd.trackIdx < 0 || cmd.trackIdx >= NUM_TRACKS) return;
        auto& track = tracks[cmd.trackIdx];
        long currentSample = globalSampleCounter.load();

        switch (cmd.type) {
            case CommandType::HandleMain:
                switch (track.state.load()) {
                    case LooperState::Ready:
                        if (track.index == 0) {
                            track.state = LooperState::Recording;
                            track.writeIndex = 0;
                            masterLoopStartSample = currentSample;
                        } else if (masterLoopDurationSamples.load() > 0) {
                            track.state = LooperState::Waiting;
                        }
                        break;
                    case LooperState::Recording:
                        track.loopDurationSamples = track.writeIndex;
                        track.state = LooperState::Playing;
                        if (track.index == 0) {
                            masterLoopDurationSamples = track.loopDurationSamples;
                            masterBPM = (60.f / ((float)masterLoopDurationSamples.load() / sampleRate)) * 4.0f;
                            beforeFxDelay.updateFromBPM(masterBPM.load());
                        }
                        break;
                    case LooperState::Playing:
                        track.state = LooperState::ArmOverdub;
                        break;
                    case LooperState::Overdub:
                        track.state = LooperState::Playing;
                        break;
                    default: break;
                }
                break;
            case CommandType::HandleStop:
                if (track.state.load() == LooperState::Playing || track.state.load() == LooperState::Overdub) {
                    track.state = LooperState::Stopped;
                } else if (track.state.load() == LooperState::Stopped) {
                    track.state = LooperState::Playing;
                    if (masterLoopDurationSamples.load() > 0) {
                        track.readIndex = (currentSample - masterLoopStartSample.load()) % masterLoopDurationSamples.load();
                    } else {
                        track.readIndex = 0;
                    }
                }
                break;
            case CommandType::Clear:
                track.clear();
                if (cmd.trackIdx == 0) {
                    masterLoopDurationSamples = 0;
                    masterBPM = 120.f;
                    beforeFxDelay.updateFromBPM(masterBPM.load());
                    for(size_t i = 1; i < tracks.size(); ++i) tracks[i].clear();
                }
                break;
            case CommandType::SetVolume:
                track.volume.setTarget(cmd.value);
                break;
            case CommandType::AddFx: {
                auto oldChainPtr = track.fxChain.load();
                auto newChainVec = std::make_shared<std::vector<std::unique_ptr<Effect>>>();
                if(oldChainPtr) {
                    for (const auto& fx : *oldChainPtr) {
                        if (fx) newChainVec->push_back(fx->clone());
                    }
                }
                switch(cmd.fxId) { // Note: fxId here is fxType
                    case 1: newChainVec->push_back(std::make_unique<RealTimePitchShifter>(track.nextFxId++)); break;
                    case 2: newChainVec->push_back(std::make_unique<Filter>(track.nextFxId++, true)); break; // Low-pass
                    case 3: newChainVec->push_back(std::make_unique<Filter>(track.nextFxId++, false)); break; // High-pass
                    case 4: newChainVec->push_back(std::make_unique<Panner>(track.nextFxId++)); break;
                    case 5: newChainVec->push_back(std::make_unique<Delay>(track.nextFxId++, "Track Delay")); break;
                    case 6: newChainVec->push_back(std::make_unique<Compressor>(track.nextFxId++)); break;
                    default: break;
                }
                for (auto& fx : *newChainVec) { if(fx) fx->setSampleRate(sampleRate); }
                track.fxChain.store(std::move(newChainVec));
                break;
            }
            case CommandType::RemoveFx: {
                auto oldChainPtr = track.fxChain.load();
                auto newChainVec = std::make_shared<std::vector<std::unique_ptr<Effect>>>();
                if(oldChainPtr){
                    for (const auto& fx : *oldChainPtr) {
                        if (fx && fx->id != cmd.fxId) {
                            newChainVec->push_back(fx->clone());
                        }
                    }
                }
                track.fxChain.store(std::move(newChainVec));
                break;
            }
            case CommandType::SetFxParam: {
                auto currentChain = track.fxChain.load();
                if(currentChain) {
                    for (const auto& fx : *currentChain) {
                        if (fx && fx->id == cmd.fxId) {
                            fx->setParam(cmd.paramIndex, cmd.value);
                        }
                    }
                }
                break;
            }
            case CommandType::ToggleBypassFx: {
                auto currentChain = track.fxChain.load();
                if(currentChain) {
                    for (const auto& fx : *currentChain) {
                        if (fx && fx->id == cmd.fxId) {
                            fx->bypass = !fx->bypass.load();
                        }
                    }
                }
                break;
            }
            case CommandType::MoveFx: {
                auto currentChain = track.fxChain.load();
                if (currentChain) {
                    for (size_t i = 0; i < currentChain->size(); ++i) {
                        if (currentChain->at(i)->id == cmd.fxId) {
                            size_t newIndex = i + (int)cmd.value;
                            if (newIndex >= 0 && newIndex < currentChain->size()) {
                                std::swap(currentChain->at(i), currentChain->at(newIndex));
                                break;
                            }
                        }
                    }
                }
                break;
            }
            default: break;
        }
    }

public:
    void handleMainButton(int trackIdx) { commandQueue.push({CommandType::HandleMain, trackIdx}); }
    void handleStopButton(int trackIdx) { commandQueue.push({CommandType::HandleStop, trackIdx}); }
    void clearTrack(int trackIdx) { commandQueue.push({CommandType::Clear, trackIdx}); }
    void setTrackVolume(int trackIdx, float vol) { commandQueue.push({CommandType::SetVolume, trackIdx, -1, -1, vol}); }
    void addEffectToTrack(int trackIdx, int fxType) { commandQueue.push({CommandType::AddFx, trackIdx, fxType}); }
    void removeEffectFromTrack(int trackIdx, int fxId) { commandQueue.push({CommandType::RemoveFx, trackIdx, fxId}); }
    void toggleEffectBypass(int trackIdx, int fxId) { commandQueue.push({CommandType::ToggleBypassFx, trackIdx, fxId}); }
    void setEffectParameter(int trackIdx, int fxId, int paramIndex, float value) {
        commandQueue.push({CommandType::SetFxParam, trackIdx, fxId, paramIndex, value});
    }
    void setLiveMonitoring(bool on) { liveMonitoringOn = on; }
    void moveEffectInTrack(int trackIdx, int fxId, int direction) { commandQueue.push({CommandType::MoveFx, trackIdx, fxId, -1, (float)direction}); }
};

// ===================================================================================
// MARK: - AAudio Backend Stub & Main Test Function
// ===================================================================================

#if __ANDROID__
static LooperEngine* engineInstance = nullptr;
static aaudio_data_callback_result_t audioCallbackTrampoline(AAudioStream* stream, void* userData, void* audioData, int32_t numFrames) {
    if(engineInstance) {
        engineInstance->process(nullptr, static_cast<float*>(audioData), numFrames);
    }
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}
void init_engine_instance(float sampleRate) {
    if(!engineInstance) {
        engineInstance = new LooperEngine(sampleRate);
    }
}
#endif // __ANDROID__