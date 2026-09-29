$inputPath = 'D:\WX\xwechat_files\wxid_6pzgwze9y27z22_e5f3\msg\file\2026-08\收入证明.doc'
$outputPath = 'D:\WX\xwechat_files\wxid_6pzgwze9y27z22_e5f3\msg\file\2026-08\收入证明.pdf'

if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Input file not found: $inputPath"
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Open($inputPath, $false, $true)
    $doc.ExportAsFixedFormat($outputPath, 17, $false, 0, 0, 0, 0, 0, $true, $false, 0, $true, $true, $false)
    Write-Output $outputPath
}
finally {
    if ($doc -ne $null) {
        $doc.Close($false)
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
    if ($word -ne $null) {
        $word.Quit()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    }
}
