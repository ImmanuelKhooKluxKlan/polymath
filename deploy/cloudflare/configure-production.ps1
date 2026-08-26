[CmdletBinding()]
param(
  [string]$AccountId = '9c8b0c2fbbe89d2705bd0a30af9c3e32',
  [string]$ZoneId = 'ce57d8d08c644851a7aaf7eb067d670c',
  [string]$ApiHostname = 'api.polymathmusician67.com',
  [string]$OhioOrigin = 'api-us-origin.polymathmusician67.com',
  [string]$SingaporeOrigin = 'api-apac-origin.polymathmusician67.com',
  [switch]$Cutover
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Read-DotEnv {
  param([Parameter(Mandatory)][string]$Path)

  $values = @{}
  if (-not (Test-Path -LiteralPath $Path)) {
    return $values
  }

  foreach ($line in Get-Content -LiteralPath $Path) {
    if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
      continue
    }

    $value = $Matches[2].Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }
    $values[$Matches[1]] = $value
  }
  return $values
}

$cloudflareEnv = Read-DotEnv -Path (Join-Path $repoRoot '.env.cloudflare.local')
$artifactEnv = Read-DotEnv -Path (Join-Path $repoRoot '.env.artifacts.local')
$apiToken = if ($env:CLOUDFLARE_API_TOKEN) {
  $env:CLOUDFLARE_API_TOKEN
} else {
  $cloudflareEnv['CLOUDFLARE_API_TOKEN']
}
$bucketName = if ($env:ARTIFACT_S3_BUCKET) {
  $env:ARTIFACT_S3_BUCKET
} else {
  $artifactEnv['ARTIFACT_S3_BUCKET']
}

if ([string]::IsNullOrWhiteSpace($apiToken)) {
  throw 'CLOUDFLARE_API_TOKEN is missing. Set it in the process environment or .env.cloudflare.local.'
}
if ([string]::IsNullOrWhiteSpace($bucketName)) {
  throw 'ARTIFACT_S3_BUCKET is missing. Set it in the process environment or .env.artifacts.local.'
}

$headers = @{
  Authorization = "Bearer $apiToken"
  'Content-Type' = 'application/json'
}
$apiBase = 'https://api.cloudflare.com/client/v4'

function Invoke-Cloudflare {
  param(
    [Parameter(Mandatory)][ValidateSet('GET', 'POST', 'PUT', 'PATCH')][string]$Method,
    [Parameter(Mandatory)][string]$Path,
    [object]$Body
  )

  $parameters = @{
    Uri = "$apiBase$Path"
    Method = $Method
    Headers = $headers
    UseBasicParsing = $true
  }
  if ($null -ne $Body) {
    $parameters.Body = $Body | ConvertTo-Json -Depth 20 -Compress
  }

  try {
    $response = Invoke-RestMethod @parameters
  } catch {
    $message = $_.Exception.Message
    if ($_.ErrorDetails.Message) {
      try {
        $details = $_.ErrorDetails.Message | ConvertFrom-Json
        if ($details.errors) {
          $message = ($details.errors | ForEach-Object { $_.message }) -join '; '
        }
      } catch {
        $message = $_.ErrorDetails.Message
      }
    }
    throw "Cloudflare $Method $Path failed: $message"
  }

  if ($response.PSObject.Properties.Name -contains 'success' -and -not $response.success) {
    $messages = ($response.errors | ForEach-Object { $_.message }) -join '; '
    throw "Cloudflare $Method $Path failed: $messages"
  }
  return $response
}

function Set-RuleById {
  param(
    [object[]]$Rules,
    [Parameter(Mandatory)][string]$Id,
    [Parameter(Mandatory)][object]$Replacement
  )

  $kept = @($Rules | Where-Object { $_.id -ne $Id })
  return @($kept) + @($Replacement)
}

function Get-OrCreateMonitor {
  $path = "/accounts/$AccountId/load_balancers/monitors"
  $description = 'Polymath API HTTPS health monitor'
  $list = Invoke-Cloudflare -Method GET -Path $path
  $existing = @($list.result) | Where-Object { $_.description -eq $description } | Select-Object -First 1
  $body = @{
    type = 'https'
    description = $description
    method = 'GET'
    path = '/api/health'
    port = 443
    expected_codes = '2xx'
    follow_redirects = $true
    allow_insecure = $false
    timeout = 5
    retries = 1
    interval = 60
    consecutive_up = 1
    consecutive_down = 2
  }

  if ($existing) {
    return (Invoke-Cloudflare -Method PUT -Path "$path/$($existing.id)" -Body $body).result
  }
  return (Invoke-Cloudflare -Method POST -Path $path -Body $body).result
}

function Get-OrCreatePool {
  param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$Description,
    [Parameter(Mandatory)][string]$OriginName,
    [Parameter(Mandatory)][string]$OriginAddress,
    [Parameter(Mandatory)][string]$MonitorId
  )

  $path = "/accounts/$AccountId/load_balancers/pools"
  $list = Invoke-Cloudflare -Method GET -Path $path
  $existing = @($list.result) | Where-Object { $_.name -eq $Name } | Select-Object -First 1
  $body = @{
    name = $Name
    description = $Description
    enabled = $true
    minimum_origins = 1
    monitor = $MonitorId
    check_regions = @('WNAM', 'ENAM', 'SEAS', 'NEAS')
    origins = @(
      @{
        name = $OriginName
        address = $OriginAddress
        enabled = $true
        weight = 1
      }
    )
  }

  if ($existing) {
    return (Invoke-Cloudflare -Method PUT -Path "$path/$($existing.id)" -Body $body).result
  }
  return (Invoke-Cloudflare -Method POST -Path $path -Body $body).result
}

Write-Host '[1/5] Verifying Cloudflare token'
$null = Invoke-Cloudflare -Method GET -Path '/user/tokens/verify'

Write-Host '[2/5] Applying direct-upload CORS without deleting unrelated rules'
$corsPath = "/accounts/$AccountId/r2/buckets/$bucketName/cors"
$corsCurrent = Invoke-Cloudflare -Method GET -Path $corsPath
$corsRule = @{
  id = 'polymath-direct-uploads'
  allowed = @{
    methods = @('PUT')
    headers = @('Content-Type')
    origins = @(
      'https://polymathmusician67.com',
      'https://www.polymathmusician67.com',
      'http://localhost:5173',
      'http://localhost:5174',
      'http://127.0.0.1:5173',
      'http://127.0.0.1:5174'
    )
  }
  exposeHeaders = @('ETag')
  maxAgeSeconds = 3600
}
$corsRules = Set-RuleById -Rules @($corsCurrent.result.rules) -Id $corsRule.id -Replacement $corsRule
$null = Invoke-Cloudflare -Method PUT -Path $corsPath -Body @{ rules = $corsRules }

Write-Host '[3/5] Applying two-day cleanup for abandoned pending uploads'
$lifecyclePath = "/accounts/$AccountId/r2/buckets/$bucketName/lifecycle"
$lifecycleCurrent = Invoke-Cloudflare -Method GET -Path $lifecyclePath
$lifecycleRule = @{
  id = 'polymath-expire-pending-uploads'
  enabled = $true
  conditions = @{ prefix = 'pending/' }
  deleteObjectsTransition = @{
    condition = @{ type = 'Age'; maxAge = 172800 }
  }
}
$lifecycleRules = Set-RuleById -Rules @($lifecycleCurrent.result.rules) -Id $lifecycleRule.id -Replacement $lifecycleRule
$null = Invoke-Cloudflare -Method PUT -Path $lifecyclePath -Body @{ rules = $lifecycleRules }

Write-Host '[4/5] Creating or updating health monitor and regional pools'
$monitor = Get-OrCreateMonitor
$ohioPool = Get-OrCreatePool -Name 'polymath-ohio' -Description 'Polymath API - AWS Ohio' -OriginName 'polymath-api-us' -OriginAddress $OhioOrigin -MonitorId $monitor.id
$singaporePool = Get-OrCreatePool -Name 'polymath-singapore' -Description 'Polymath API - AWS Singapore' -OriginName 'polymath-api-apac' -OriginAddress $SingaporeOrigin -MonitorId $monitor.id

if (-not $Cutover) {
  Write-Host '[5/5] Staged safely. Production DNS was not changed.'
  Write-Host 'Re-run with -Cutover only after both pools report healthy.'
  return
}

Write-Host '[5/5] Checking pool health before production cutover'
$ohioHealth = Invoke-Cloudflare -Method GET -Path "/accounts/$AccountId/load_balancers/pools/$($ohioPool.id)/health"
$singaporeHealth = Invoke-Cloudflare -Method GET -Path "/accounts/$AccountId/load_balancers/pools/$($singaporePool.id)/health"
$ohioHealthy = $ohioHealth.result.pop_health.healthy -eq $true
$singaporeHealthy = $singaporeHealth.result.pop_health.healthy -eq $true
if (-not $ohioHealthy -or -not $singaporeHealthy) {
  throw 'Cutover stopped: both Cloudflare pools must report healthy first.'
}

$loadBalancerPath = "/zones/$ZoneId/load_balancers"
$loadBalancers = Invoke-Cloudflare -Method GET -Path $loadBalancerPath
$existingLoadBalancer = @($loadBalancers.result) | Where-Object { $_.name -eq $ApiHostname } | Select-Object -First 1
$loadBalancerBody = @{
  name = $ApiHostname
  description = 'Polymath API regional routing and automatic failover'
  enabled = $true
  proxied = $true
  ttl = 30
  steering_policy = 'geo'
  default_pools = @($ohioPool.id, $singaporePool.id)
  fallback_pool = $ohioPool.id
  region_pools = @{
    WNAM = @($ohioPool.id, $singaporePool.id)
    ENAM = @($ohioPool.id, $singaporePool.id)
    NSAM = @($ohioPool.id, $singaporePool.id)
    SSAM = @($ohioPool.id, $singaporePool.id)
    SEAS = @($singaporePool.id, $ohioPool.id)
    NEAS = @($singaporePool.id, $ohioPool.id)
    SAS = @($singaporePool.id, $ohioPool.id)
    OC = @($singaporePool.id, $ohioPool.id)
  }
  session_affinity = 'cookie'
  session_affinity_ttl = 1800
  adaptive_routing = @{ failover_across_pools = $true }
}

if ($existingLoadBalancer) {
  $null = Invoke-Cloudflare -Method PUT -Path "$loadBalancerPath/$($existingLoadBalancer.id)" -Body $loadBalancerBody
} else {
  $null = Invoke-Cloudflare -Method POST -Path $loadBalancerPath -Body $loadBalancerBody
}

Write-Host 'Production Cloudflare load balancing is enabled.'
