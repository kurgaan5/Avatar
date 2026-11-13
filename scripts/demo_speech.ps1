param(
    [string]$Text = (Join-Path $PSScriptRoot '../examples/demo/speech.txt'),
    [string]$Output = (Join-Path $PSScriptRoot '../output/demo-speech.wav'),
    [string]$Culture = 'ru-RU'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$voice = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $installed = $voice.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -eq $Culture
    } | Select-Object -First 1
    if (-not $installed) { throw "Install a Windows speech voice for $Culture." }
    $voice.SelectVoice($installed.VoiceInfo.Name)
    $path = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
    $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
        16000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono
    )
    $voice.SetOutputToWaveFile($path, $format)
    $voice.Speak((Get-Content -Raw -Encoding UTF8 -LiteralPath $Text))
    $path
}
finally {
    $voice.Dispose()
}
