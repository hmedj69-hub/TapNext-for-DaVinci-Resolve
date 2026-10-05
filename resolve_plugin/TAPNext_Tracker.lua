--[[
TAPNext++ Tracker — intégration DaVinci Resolve
================================================

Workspace > Scripts > TAPNext_Tracker (toutes les pages).

  • Lancé sur un clip : ouvre TAPNext Studio sur ce clip (partie utilisée
    dans la timeline). Toute l'interface est dans Studio.
  • Après « Exporter et envoyer à Resolve » dans Studio, les résultats sont
    importés : matte attachée au clip (page Color → Add Matte), nœud Fusion
    ajouté dans la comp du clip si demandé.
      - automatiquement si Resolve fournit la fenêtre d'état (UIManager) ;
      - sinon en relançant simplement ce script une fois l'export terminé.
  • Ne dépend pas de UIManager pour fonctionner. Les erreurs s'affichent dans
    une boîte de dialogue et dans <dossier outil>/jobs/resolve_log.txt.
]]

local TAPNEXT_ROOT = [[@@TAPNEXT_ROOT@@]]

local M = {}

-- ---------------------------------------------------------------- utilitaires
M.IS_WIN = package.config:sub(1, 1) == "\\"
M.SEP = M.IS_WIN and "\\" or "/"

function M.join(...)
  local parts = { ... }
  local out = table.concat(parts, M.SEP)
  out = out:gsub("[/\\]+", M.SEP)
  return out
end

function M.dirname(p)
  return (p:match("^(.*)[/\\][^/\\]*$")) or "."
end

function M.basename_noext(p)
  local b = p:match("([^/\\]+)$") or p
  return (b:gsub("%.[^%.]+$", ""))
end

function M.exists(p)
  local f = io.open(p, "rb")
  if f then f:close() return true end
  return false
end

function M.read_all(p)
  local f = io.open(p, "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  return s
end

function M.write_all(p, s)
  local f, err = io.open(p, "wb")
  if not f then return false, err end
  f:write(s)
  f:close()
  return true
end

function M.can_write(p)
  local probe = M.join(p, ".tapnext_write_test")
  local ok = M.write_all(probe, "ok")
  if ok then os.remove(probe) end
  return ok
end

--[[ Crée un dossier sans ouvrir de console : bmd.createdir (API Fusion) ;
     la commande système n'est qu'un dernier recours. ]]
function M.mkdir(p)
  if M.can_write(p) then return true end
  local b = rawget(_G, "bmd")
  if b and b.createdir then
    pcall(b.createdir, p)
    if M.can_write(p) then return true end
  end
  if M.IS_WIN then
    os.execute('if not exist "' .. p .. '" mkdir "' .. p .. '"')
  else
    os.execute("mkdir -p '" .. p:gsub("'", "'\\''") .. "'")
  end
  return M.can_write(p)
end

function M.dir_writable(p)
  return M.mkdir(p)
end

-- Sous Windows, le Lua de Resolve ouvre les fichiers et lance les commandes
-- avec l'encodage ANSI : on évite donc les chemins non ASCII pour ses propres
-- fichiers (le chemin du média passe, lui, par le JSON en UTF-8).
function M.is_ascii(s)
  return not tostring(s):find("[\128-\255]")
end

function M.root_is_valid(root)
  return root and root ~= "" and not root:find("@@", 1, true)
    and M.exists(M.join(root, "tap_resolve_tool.py"))
end

function M.python_exe(root)
  if M.IS_WIN then return M.join(root, ".venv", "Scripts", "python.exe") end
  return M.join(root, ".venv", "bin", "python")
end

-- Échappement d'un argument pour un .bat (in_bat = true : « % » doublé) ou sh.
function M.quote(arg, in_bat)
  arg = tostring(arg)
  if M.IS_WIN then
    if in_bat then arg = arg:gsub("%%", "%%%%") end
    return '"' .. arg .. '"'
  end
  return "'" .. arg:gsub("'", "'\\''") .. "'"
end

function M.fmt_num(v)
  if math.floor(v) == v then return string.format("%d", v) end
  return (string.format("%.4f", v):gsub("0+$", ""):gsub("%.$", ""))
end

-- ------------------------------------------------------------- journal & messages
M.LOG_LINES = {}

function M.log(root, msg)
  print("[TAPNext] " .. msg)
  M.LOG_LINES[#M.LOG_LINES + 1] = os.date("%H:%M:%S ") .. msg
  if M.root_is_valid(root) then
    local f = io.open(M.join(root, "jobs", "resolve_log.txt"), "ab")
    if f then f:write(os.date("%Y-%m-%d %H:%M:%S ") .. msg .. "\n"); f:close() end
  end
end

--[[ Boîte de dialogue native du système (ne dépend pas de UIManager, absent
     de certaines éditions de Resolve). Le texte passe par un fichier UTF-8
     pour conserver les accents. ]]
function M.ui_message(text)
  local f = rawget(_G, "fu") or rawget(_G, "fusion")
  local ui = f and f.UIManager
  local b = rawget(_G, "bmd")
  if not (ui and b and b.UIDispatcher) then return false end
  return pcall(function()
    local disp = b.UIDispatcher(ui)
    local win = disp:AddWindow({ ID = "TAPMsg", WindowTitle = "TAPNext++",
      Geometry = { 320, 260, 560, 240 } },
      ui:VGroup {
        ui:Label { ID = "T", Text = text, WordWrap = true, Weight = 1 },
        ui:HGroup { Weight = 0, ui:HGap(0, 1), ui:Button { ID = "Ok", Text = "OK" } },
      })
    local function close() disp:ExitLoop() end
    win.On.Ok.Clicked = close
    win.On.TAPMsg.Close = close
    win:Show()
    disp:RunLoop()
    win:Hide()
  end)
end

function M.message(root, text)
  print("[TAPNext] " .. text)
  if M.ui_message(text) then return end
  local dir = M.root_is_valid(root) and M.join(root, "jobs") or nil
  if dir then M.mkdir(dir) end
  local file = dir and M.join(dir, "message.txt")
  if not (file and M.write_all(file, text)) then return end
  if M.IS_WIN then
    if not M.is_ascii(file) then return end
    os.execute('powershell -NoProfile -WindowStyle Hidden -Command "Add-Type -AssemblyName ' ..
      'PresentationFramework; $t = Get-Content -Raw -Encoding UTF8 \'' .. file .. '\'; ' ..
      '[void][System.Windows.MessageBox]::Show($t, \'TAPNext++\')"')
  elseif io.popen("uname"):read("*l") == "Darwin" then
    os.execute("osascript -e 'display dialog (read POSIX file \"" .. file ..
      "\" as «class utf8») with title \"TAPNext++\" buttons {\"OK\"} default button 1' >/dev/null 2>&1")
  else
    os.execute("zenity --info --title=TAPNext++ --text=\"$(cat " .. M.quote(file) .. ")\" >/dev/null 2>&1 &")
  end
end

-- ------------------------------------------------------------- tâche Studio
function M.json_escape(v)
  return (tostring(v):gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n"))
end

function M.json_field(text, key)
  if not text then return nil end
  local k = '"' .. key .. '"%s*:%s*'
  if text:match(k .. '""') then return "" end
  local v = text:match(k .. '"(.-[^\\])"')
  if v then return (v:gsub("\\/", "/"):gsub('\\"', '"'):gsub("\\n", "\n"):gsub("\\\\", "\\")) end
  return text:match(k .. "(%-?[%d%.]+)") or text:match(k .. "(%a+)")
end

function M.write_json(path, tbl)
  local keys = {}
  for k in pairs(tbl) do keys[#keys + 1] = k end
  table.sort(keys)
  local parts = {}
  for _, k in ipairs(keys) do
    local v = tbl[k]
    if type(v) == "number" then
      parts[#parts + 1] = string.format('"%s": %s', k, M.fmt_num(v))
    elseif type(v) == "boolean" then
      parts[#parts + 1] = string.format('"%s": %s', k, tostring(v))
    else
      parts[#parts + 1] = string.format('"%s": "%s"', k, M.json_escape(v))
    end
  end
  return M.write_all(path, "{" .. table.concat(parts, ", ") .. "}")
end

function M.studio_cmd(root, job_path)
  if M.IS_WIN then
    local pyw = M.join(root, ".venv", "Scripts", "pythonw.exe")
    if not M.exists(pyw) then pyw = M.python_exe(root) end
    return 'start "" ' .. M.quote(pyw) .. " " .. M.quote(M.join(root, "tap_studio.py"))
      .. " --job " .. M.quote(job_path)
  end
  return M.quote(M.python_exe(root)) .. " " .. M.quote(M.join(root, "tap_studio.py"))
    .. " --job " .. M.quote(job_path) .. " >/dev/null 2>&1 &"
end

function M.sleep(sec)
  local b = rawget(_G, "bmd")
  if b and b.wait then
    if pcall(b.wait, sec) then return end
  end
  local t = os.clock() + sec
  while os.clock() < t do end
end

--[[ Lance Studio sans fenêtre de console (bmd.executebg + pythonw). Studio
     écrit jobs/resolve_started.txt dès son démarrage : sans ce signal après
     quelques secondes, on se rabat sur la commande système classique. ]]
function M.launch_studio(root, job_path)
  local started = M.join(M.dirname(job_path), "resolve_started.txt")
  os.remove(started)
  local b = rawget(_G, "bmd")
  if M.IS_WIN and b and b.executebg then
    local pyw = M.join(root, ".venv", "Scripts", "pythonw.exe")
    if M.exists(pyw) then
      local ok = pcall(b.executebg, M.quote(pyw) .. " " .. M.quote(M.join(root, "tap_studio.py"))
        .. " --job " .. M.quote(job_path))
      if ok then
        for _ = 1, 40 do
          if M.exists(started) then return end
          M.sleep(0.25)
        end
        M.log(root, "bmd.executebg sans effet : lancement classique.")
      end
    end
  end
  os.execute(M.studio_cmd(root, job_path))
end

function M.paths(root)
  local jobs = M.join(root, "jobs")
  return {
    dir = jobs,
    job = M.join(jobs, "resolve_job.json"),
    done = M.join(jobs, "resolve_done.json"),
    ack = M.join(jobs, "resolve_ack.json"),
  }
end

-- ------------------------------------------------------------- Resolve
function M.get_context(resolve)
  local pm = resolve:GetProjectManager()
  local proj = pm and pm:GetCurrentProject()
  if not proj then return nil, "Aucun projet ouvert." end
  local tl = proj:GetCurrentTimeline()
  if not tl then return nil, "Aucune timeline active." end
  local item = tl:GetCurrentVideoItem()
  if not item then
    return nil, "Aucun clip vidéo sous la tête de lecture.\n" ..
      "Placez la tête de lecture sur le clip à traiter."
  end
  local mpi = item:GetMediaPoolItem()
  if not mpi then return nil, "Ce clip n'a pas de média source (générateur, titre…)." end
  local path = mpi:GetClipProperty("File Path")
  if not path or path == "" then return nil, "Chemin du média introuvable." end
  local left = tonumber(item:GetLeftOffset()) or 0
  local dur = tonumber(item:GetDuration()) or 0
  return {
    project = proj, timeline = tl, item = item, mpi = mpi, path = path,
    name = item:GetName() or M.basename_noext(path),
    in_frame = math.floor(left),
    out_frame = math.floor(left + math.max(dur, 1) - 1),
  }
end

function M.attach_matte(resolve, ctx, matte_path)
  local ms = resolve:GetMediaStorage()
  if not ms then return false end
  local ok, res = pcall(function()
    return ms:AddClipMattesToMediaPool(ctx.mpi, { matte_path })
  end)
  return ok and res and true or false
end

--[[ Importe un fichier dans le chutier « TAPNext » du Media Pool (créé si
     besoin). Renvoie true (ou true, "déjà là") si l'élément est dans le chutier,
     sinon false et la raison. ]]
function M.norm_path(p)
  return (tostring(p or ""):gsub("\\", "/"):lower())
end

function M.import_to_bin(resolve, path)
  local reason = "raison inconnue"
  local ok, res, extra = pcall(function()
    local proj = resolve:GetProjectManager():GetCurrentProject()
    if not proj then return false, "aucun projet ouvert" end
    local mp = proj:GetMediaPool()
    local rootf = mp:GetRootFolder()
    local bin
    for _, f in pairs(rootf:GetSubFolderList() or {}) do
      if f:GetName() == "TAPNext" then bin = f end
    end
    if not bin then bin = mp:AddSubFolder(rootf, "TAPNext") end
    bin = bin or rootf
    -- déjà présent (même fichier) ?
    local want = M.norm_path(path)
    for _, c in pairs(bin:GetClipList() or {}) do
      local okp, fp = pcall(function() return c:GetClipProperty("File Path") end)
      if okp and M.norm_path(fp) == want then return true, "déjà dans le chutier" end
    end
    local prev = mp:GetCurrentFolder()
    mp:SetCurrentFolder(bin)
    local n = 0
    local items = mp:ImportMedia({ path })
    for _ in pairs(items or {}) do n = n + 1 end
    if n == 0 then
      -- Seconde méthode : via le stockage de médias (dossier courant = chutier).
      local ms = resolve:GetMediaStorage()
      local items2 = ms and ms:AddItemListToMediaPool({ path })
      for _ in pairs(items2 or {}) do n = n + 1 end
    end
    if prev then mp:SetCurrentFolder(prev) end
    if n == 0 then
      return false, M.exists(path) and "Resolve refuse le fichier" or "fichier introuvable"
    end
    return true
  end)
  if not ok then return false, tostring(res) end
  if res then return true, extra end
  return false, extra or reason
end

--[[ Comp Fusion du clip (créée si besoin) + décalage source → comp. ]]
function M.get_comp(ctx)
  local item = ctx.item
  local comp
  if (item:GetFusionCompCount() or 0) > 0 then
    comp = item:GetFusionCompByIndex(1)
  else
    comp = item:AddFusionComp()
  end
  if not comp then return nil, 0 end
  local attrs = comp:GetAttrs() or {}
  return comp, math.floor(tonumber(attrs.COMPN_RenderStart) or 0)
end

--[[ Les .setting de Studio ont l'image 0 = point d'entrée du clip ; on les
     décale de RenderStart (souvent 0, parfois 1001). ]]
function M.shift_setting(text, delta)
  if not text or delta == 0 then return text end
  return (text:gsub("%[(%-?%d+)%] = {", function(n)
    return "[" .. (tonumber(n) + delta) .. "] = {"
  end))
end

--[[ Construit les nœuds avec l'API (comp:AddTool) à partir du fichier
     « _tools.lua » écrit par Studio. Ne dépend pas de l'affichage de la comp. ]]
function M.load_tools(path)
  local text = path and M.read_all(path)
  if not text then return nil end
  local loader = rawget(_G, "loadstring") or load
  local fn = loader(text)
  if not fn then return nil end
  local ok, specs = pcall(fn)
  return ok and type(specs) == "table" and specs or nil
end

function M.build_tools(comp, specs, delta)
  local made = {}
  for _, sp in ipairs(specs) do
    local t = comp:AddTool(sp.type, -32768, -32768)
    if not t then error("AddTool(" .. tostring(sp.type) .. ") a échoué") end
    pcall(function() t:SetAttrs({ TOOLS_Name = sp.name }) end)
    for k, v in pairs(sp.static or {}) do
      pcall(function() t:SetInput(k, v) end)
    end
    for k, keys in pairs(sp.numbers or {}) do
      t[k] = comp:BezierSpline()
      local inp = t[k]
      for f, v in pairs(keys) do inp[f + delta] = v end
    end
    for k, keys in pairs(sp.points or {}) do
      t[k] = comp:XYPath()
      local inp = t[k]
      for f, v in pairs(keys) do inp[f + delta] = { v[1], v[2] } end
    end
    made[#made + 1] = t
  end
  return made
end

function M.wire_before_output(comp, tool)
  local mo = (comp:GetToolList(false, "MediaOut") or {})[1]
  if not (tool and mo) then return nil end
  local src = mo.Input and mo.Input:GetConnectedOutput()
  if src then
    tool:ConnectInput("Input", src:GetTool())
  else
    local mi = (comp:GetToolList(false, "MediaIn") or {})[1]
    if mi then tool:ConnectInput("Input", mi) end
  end
  mo:ConnectInput("Input", tool)
  return "Stabilisation branchée avant " .. (mo.Name or "MediaOut") .. " : l'image est stabilisée."
end

--[[ Ajoute les nœuds d'un .setting dans la comp : collage (comp:Paste), puis,
     s'il échoue, construction par l'API. Renvoie ok, message, méthode. ]]
function M.paste_into_comp(comp, setting_text, insert_before_output, tools_path, delta)
  local specs = M.load_tools(tools_path)
  local main = specs and specs[1] and specs[1].name
  local function count_main()
    if not main then return 0 end
    local n = 0
    for _, t in pairs(comp:GetToolList(false) or {}) do
      local nm = t.Name or ""
      if nm == main or nm:find("^" .. main .. "_%d+$") then n = n + 1 end
    end
    return n
  end
  local before = count_main()
  comp:Lock()
  comp:StartUndo("TAPNext++")
  local how, err = nil, nil
  local data = setting_text and bmd.readstring(setting_text)
  if data then
    local okp, res = pcall(function() return comp:Paste(data) end)
    if okp and res and (not main or count_main() > before) then how = "collage" end
    err = okp and "comp:Paste refusé" or tostring(res)
  else
    err = "fichier .setting illisible"
  end
  local tool
  if not how and not specs then err = err .. ", fichier _tools.lua absent" end
  if not how and specs then
    local okb, made = pcall(M.build_tools, comp, specs, delta or 0)
    if okb then
      how, tool = "api", made[1]
    else
      err = tostring(made)
    end
  end
  local msg = how and "Nœud ajouté dans la comp Fusion." or ("échec : " .. tostring(err))
  if how and insert_before_output then
    if not tool then
      for _, t in pairs(comp:GetToolList(true) or {}) do
        if (t.Name or ""):find("^TAP_Stabilize") then tool = t end
      end
    end
    msg = M.wire_before_output(comp, tool) or msg
  end
  comp:EndUndo(true)
  comp:Unlock()
  return how ~= nil, msg, how
end

-- ------------------------------------------------------------- Resolve : accès
function M.get_resolve()
  local r = rawget(_G, "resolve")
  if r then return r end
  local fn = rawget(_G, "Resolve")
  if type(fn) == "function" then
    local ok, v = pcall(fn)
    if ok and v then return v end
  end
  local b = rawget(_G, "bmd")
  if b and b.scriptapp then
    local ok, v = pcall(b.scriptapp, "Resolve")
    if ok and v then return v end
  end
  local f = rawget(_G, "fu") or rawget(_G, "fusion")
  if f then
    local ok, v = pcall(function() return f:GetResolve() end)
    if ok and v then return v end
  end
  return nil
end

--[[ Retrouve le clip de la timeline correspondant à une tâche (identifiant
     unique, sinon même média sous la tête de lecture). ]]
function M.find_item(resolve, job_text)
  local proj = resolve:GetProjectManager():GetCurrentProject()
  local tl = proj and proj:GetCurrentTimeline()
  if not tl then return nil end
  local uid = M.json_field(job_text, "item_uid")
  if uid and uid ~= "" then
    for t = 1, (tonumber(tl:GetTrackCount("video")) or 0) do
      for _, it in ipairs(tl:GetItemListInTrack("video", t) or {}) do
        local ok, id = pcall(function() return it:GetUniqueId() end)
        if ok and id == uid then return it end
      end
    end
  end
  local cur = tl:GetCurrentVideoItem()
  local mpi = cur and cur:GetMediaPoolItem()
  if mpi and mpi:GetClipProperty("File Path") == M.json_field(job_text, "video") then
    return cur
  end
  return nil
end

--[[ Importe les résultats d'un export Studio. Renvoie la liste des lignes du
     rapport (aussi écrite dans resolve_ack.json, que Studio affiche). ]]
function M.import_results(resolve, root, job_text, done_text)
  local P = M.paths(root)
  local report = {}
  local item = M.find_item(resolve, job_text)
  if not item then
    report[#report + 1] = "Clip introuvable dans la timeline active : placez la tête de lecture " ..
      "sur le clip traité puis relancez Workspace > Scripts > TAPNext_Tracker."
    return report, false
  end
  local ctx = {
    item = item, mpi = item:GetMediaPoolItem(),
    in_frame = tonumber(M.json_field(job_text, "start")) or 0,
  }
  local matte = M.json_field(done_text, "matte")
  if matte and matte ~= "" then
    local okm, why = M.import_to_bin(resolve, matte)
    if okm then
      report[#report + 1] = "✔ Matte dans le Media Pool, chutier « TAPNext »" ..
        (why and (" (" .. why .. ")") or "") .. "."
    else
      report[#report + 1] = "✘ Import de la matte dans le Media Pool impossible (" .. tostring(why) ..
        ") : glissez le fichier ci-dessous dans le Media Pool."
    end
    if M.json_field(done_text, "attach") ~= "false" and M.attach_matte(resolve, ctx, matte) then
      report[#report + 1] = "✔ Matte aussi attachée au clip (page Color → clic droit → Add Matte)."
    end
    report[#report + 1] = "   Fichier : " .. matte
    report[#report + 1] = "→ Page Color : glissez la matte du chutier TAPNext dans la zone des " ..
      "nœuds (ou clic droit → Add Matte), puis reliez sa sortie bleue (Key) à l'entrée Key " ..
      "du nœud à corriger."
  end
  local stab = M.json_field(done_text, "stabilized_video")
  if stab and stab ~= "" then
    local oks, why = M.import_to_bin(resolve, stab)
    if oks then
      report[#report + 1] = "✔ Vidéo stabilisée dans le chutier « TAPNext » (même durée et timecode " ..
        "que l'original). Page Edit : glissez-la sur le clip d'origine dans le viewer → Replace : " ..
        "elle se cale image pour image."
    else
      report[#report + 1] = "✘ Import de la vidéo stabilisée impossible (" .. tostring(why) .. ") : " .. stab
    end
  end
  local comp_v = M.json_field(done_text, "composite_video")
  if comp_v and comp_v ~= "" then
    local okc, why = M.import_to_bin(resolve, comp_v)
    if okc then
      report[#report + 1] = "✔ Rush masqué + effets dans le chutier « TAPNext » (alpha si fond " ..
        "transparent) : posez-le sur une piste au-dessus du plan."
    else
      report[#report + 1] = "✘ Import du rush masqué impossible (" .. tostring(why) .. ") : " .. comp_v
    end
  end
  local tapfx = M.json_field(done_text, "tapfx")
  if tapfx and tapfx ~= "" then
    report[#report + 1] = "✔ Suivi prêt pour l'effet OFX « TAPNext Shapes » : Effets → OpenFX → " ..
      "TAPNext → TAPNext Shapes (posé sur un nœud de la page Color ou sur le clip). " ..
      "Le fichier de suivi est rempli automatiquement ; réglez les formes dans l'Inspecteur " ..
      "(Sortie « Rush masqué + effets » pour l'écho, le slit-scan et l'ombre sur l'image)."
  end
  local labels = { stabilize = "TAP_Stabilize (stabilisation)", matchmove = "TAP_MatchMove (match-move)",
    cornerpin = "TAP_CornerPin (insertion 4 coins)",
    camera3d = "TAP_Camera3D (caméra 3D + repères TAP_Point3D)" }
  local wanted = {}
  for _, md in ipairs({ "stabilize", "matchmove", "cornerpin", "camera3d" }) do
    local f = M.json_field(done_text, "setting_" .. md)
    if f and f ~= "" then wanted[#wanted + 1] = { md, f } end
  end
  if #wanted > 0 then
    local comp, rstart = M.get_comp(ctx)
    if comp then
      local wire = M.json_field(done_text, "wire_stabilize") == "true"
      for _, w in ipairs(wanted) do
        local md, file = w[1], w[2]
        local text = M.shift_setting(M.read_all(file), rstart)
        local tools = file:gsub("%.setting$", "_tools.lua")
        if (text and text ~= "") or M.exists(tools) then
          local insert = (md == "stabilize") and wire
          local ok, msg, how = M.paste_into_comp(comp, text, insert, tools, rstart)
          M.log(root, "Nœud " .. md .. " : " .. tostring(how) .. " · " .. tostring(msg))
          if ok and not insert then msg = "Nœud " .. labels[md] .. " ajouté (à brancher)." end
          report[#report + 1] = ok and ("✔ " .. msg) or
            ("✘ Ajout impossible : " .. labels[md] .. " (" .. tostring(msg) .. ")")
        else
          report[#report + 1] = "✘ Nœud " .. labels[md] .. " illisible."
        end
      end
      report[#report + 1] = "→ Page Fusion : la comp du clip contient les nœuds TAP_…"
      for _, w in ipairs(wanted) do
        if w[1] == "camera3d" then
          report[#report + 1] = "→ Caméra 3D : reliez TAP_Camera3D et vos objets 3D à un Merge3D, " ..
            "puis à un Renderer3D ; les TAP_Point3D marquent des points réels de la scène."
        end
      end
    else
      report[#report + 1] = "✘ Comp Fusion introuvable pour ce clip."
    end
  end
  return report, true
end

function M.finish_import(resolve, root, job_text, done_text)
  local P = M.paths(root)
  local report, ok = M.import_results(resolve, root, job_text, done_text)
  M.write_json(P.ack, { status = ok and "ok" or "error", report = table.concat(report, "\n"),
    job_id = M.json_field(job_text, "job_id") or "" })
  os.remove(P.done .. ".imported")
  if not os.rename(P.done, P.done .. ".imported") then os.remove(P.done) end
  for _, l in ipairs(report) do M.log(root, l) end
  return report, ok
end

-- ------------------------------------------------------------- point d'entrée
function M.main()
  local root = TAPNEXT_ROOT
  if not M.root_is_valid(root) then
    M.message(nil, "TAPNext++ : dossier de l'outil introuvable (" .. tostring(root) .. ").\n" ..
      "Relancez INSTALLER_Windows.bat depuis le dossier TAPNext.")
    return
  end
  local P = M.paths(root)
  M.mkdir(P.dir)
  M.log(root, "Lancement du script (" .. _VERSION .. ")")
  if M.IS_WIN and not M.is_ascii(root) then
    M.message(root, "Le dossier de l'outil contient des caractères accentués :\n" .. root ..
      "\n\nDéplacez-le (ex. C:\\TAPNext) puis relancez INSTALLER_Windows.bat.")
    return
  end
  if not M.exists(M.python_exe(root)) then
    M.message(root, "Installation incomplète : lancez INSTALLER_Windows.bat dans\n" .. root)
    return
  end
  local resolve = M.get_resolve()
  if not resolve then
    M.message(root, "Impossible d'accéder à DaVinci Resolve depuis le script.\n" ..
      "Lancez-le depuis Workspace > Scripts dans Resolve.")
    return
  end

  -- 1) Un export Studio attend d'être importé ? → on l'importe.
  local done_text = M.read_all(P.done)
  local job_text = M.read_all(P.job)
  local ack_text = M.read_all(P.ack)
  if done_text and ack_text and M.json_field(ack_text, "job_id") ~= nil
      and M.json_field(ack_text, "job_id") == M.json_field(done_text, "job_id") then
    -- Déjà importé : on ne recommence pas, on passe à un nouveau suivi.
    os.remove(P.done)
    done_text = nil
  end
  if done_text and job_text and M.json_field(done_text, "status") == "ok" then
    local report, ok = M.finish_import(resolve, root, job_text, done_text)
    M.message(root, "TAPNext++ — import dans Resolve\n\n" .. table.concat(report, "\n"))
    if ok then return end
  end

  -- 2) Sinon : ouvrir TAPNext Studio sur le clip sous la tête de lecture.
  local ctx, err = M.get_context(resolve)
  if not ctx then
    M.message(root, "TAPNext++ : " .. err)
    return
  end
  local out_dir = M.join(M.dirname(ctx.path), "TAPNext")
  if (M.IS_WIN and not M.is_ascii(out_dir)) or not M.dir_writable(out_dir) then
    out_dir = M.join(root, "output")
    M.mkdir(out_dir)
  end
  local base = M.basename_noext(ctx.path)
  if M.IS_WIN and not M.is_ascii(base) then base = "clip" end
  local name = base .. "_" .. ctx.in_frame .. "-" .. ctx.out_frame
  local okid, uid = pcall(function() return ctx.item:GetUniqueId() end)
  local job_id = tostring(os.time())
  os.remove(P.done)
  os.remove(P.ack)
  M.write_json(P.job, {
    video = ctx.path, out_dir = out_dir, name = name, done = P.done, ack = P.ack,
    start = ctx.in_frame, ["end"] = ctx.out_frame, item_uid = okid and uid or "",
    clip_name = ctx.name, job_id = job_id,
  })
  M.log(root, "Ouverture de TAPNext Studio : " .. ctx.path .. " [" .. ctx.in_frame .. "-" .. ctx.out_frame .. "]")
  M.launch_studio(root, P.job)

  -- 3) Attente de l'export (si la fenêtre d'état est disponible) ; sinon
  --    l'utilisateur relance le script après l'export pour importer.
  local fusion = rawget(_G, "fu") or rawget(_G, "fusion")
  local ui = fusion and fusion.UIManager
  local b = rawget(_G, "bmd")
  if not (ui and b and b.UIDispatcher) then
    M.log(root, "UIManager indisponible : import au prochain lancement du script.")
    return
  end
  local disp = b.UIDispatcher(ui)
  local win = disp:AddWindow({ ID = "TAPWin", WindowTitle = "TAPNext++", Geometry = { 260, 200, 480, 200 } },
    ui:VGroup {
      ui:Label { ID = "Status", WordWrap = true, Weight = 1,
        Text = "TAPNext Studio est ouvert sur « " .. ctx.name .. " ».\n" ..
          "Les résultats seront importés ici automatiquement après l'export." },
      ui:HGroup { Weight = 0,
        ui:Button { ID = "Import", Text = "Importer maintenant" },
        ui:HGap(0, 1),
        ui:Button { ID = "Close", Text = "Fermer" } },
    })
  local itm = win:GetItems()
  local state = { done = false }

  -- Vérifie l'export ; renvoie true quand c'est terminé (importé ou annulé).
  local function check(manual)
    if state.done then return true end
    local d = M.read_all(P.done)
    if not d or d == "" then
      if manual then itm.Status.Text = "Pas encore d'export : terminez dans TAPNext Studio " ..
        "(« Exporter et envoyer à Resolve »)." end
      return false
    end
    state.done = true
    if M.json_field(d, "status") == "ok" then
      local report = M.finish_import(resolve, root, M.read_all(P.job), d)
      itm.Status.Text = table.concat(report, "\n")
    else
      itm.Status.Text = "TAPNext Studio a été fermé sans export."
      os.remove(P.done)
    end
    return true
  end
  local function safe_check(manual)
    local ok, res = pcall(check, manual)
    if not ok then
      state.done = true
      M.log(root, "ERREUR import : " .. tostring(res))
      itm.Status.Text = "Erreur pendant l'import : " .. tostring(res)
      return true
    end
    return res
  end

  -- Minuteur : l'événement Timeout est reçu par le dispatcher (disp.On.Timeout).
  local timer
  local ok_timer = pcall(function()
    timer = ui:Timer { ID = "TAPPoll", Interval = 1000, SingleShot = false }
    disp.On.Timeout = function()
      if safe_check(false) and timer then timer:Stop() end
    end
    timer:Start()
  end)
  if not ok_timer then
    timer = nil
    itm.Status.Text = itm.Status.Text .. "\n(Après l'export, cliquez sur « Importer maintenant ».)"
  end

  win.On.Import.Clicked = function() safe_check(true) end
  local function close()
    if timer then pcall(function() timer:Stop() end) end
    disp:ExitLoop()
  end
  win.On.Close.Clicked = close
  win.On.TAPWin.Close = close
  win:Show()
  disp:RunLoop()
  win:Hide()
end

if not TAPNEXT_TEST then
  local ok, err = pcall(M.main)
  if not ok then
    local root = TAPNEXT_ROOT
    M.log(root, "ERREUR : " .. tostring(err))
    M.message(root, "TAPNext++ : erreur inattendue\n\n" .. tostring(err) ..
      "\n\nJournal : " .. M.join(tostring(root), "jobs", "resolve_log.txt"))
  end
end

return M
