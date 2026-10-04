--[[
TAPNext++ Tracker — intégration DaVinci Resolve
================================================

Script Resolve (Lua natif) disponible dans Workspace > Scripts > TAPNext_Tracker
sur toutes les pages (dossier « Scripts/Utility »).

  1. Placez la tête de lecture sur un clip de la timeline et lancez le script.
  2. « Ouvrir dans TAPNext Studio » : le clip s'ouvre dans l'application de
     tracking, limité à la partie utilisée dans la timeline.
  3. Dans Studio : entourez le sujet, lancez le suivi, réglez la matte en
     direct, puis « Exporter et envoyer à Resolve ».
  4. De retour ici, automatiquement :
       • la matte N&B est attachée au clip (page Color → Add Matte) ;
       • option : nœud Transform (stabilisation / match-move) ajouté dans la
         comp Fusion du clip, images-clés calées sur la numérotation de la comp.

Installé par INSTALLER_Windows.bat / install.sh (le chemin du dossier de
l'outil est inscrit ci-dessous).
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

function M.mkdir(p)
  if M.IS_WIN then
    os.execute('if not exist "' .. p .. '" mkdir "' .. p .. '"')
  else
    os.execute("mkdir -p '" .. p:gsub("'", "'\\''") .. "'")
  end
end

function M.dir_writable(p)
  M.mkdir(p)
  local probe = M.join(p, ".tapnext_write_test")
  local ok = M.write_all(probe, "ok")
  if ok then os.remove(probe) end
  return ok
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

-- ------------------------------------------------------- tâche Studio
function M.json_escape(v)
  return (tostring(v):gsub("\\", "\\\\"):gsub('"', '\\"'))
end

function M.json_field(text, key)
  if not text then return nil end
  local v = text:match('"' .. key .. '"%s*:%s*"(.-)"')
  if v then return (v:gsub("\\/", "/"):gsub("\\\\", "\\")) end
  return text:match('"' .. key .. '"%s*:%s*(%-?%d+)')
end

function M.write_job(job_path, job)
  local parts = {}
  for _, k in ipairs({ "video", "out_dir", "name", "done" }) do
    parts[#parts + 1] = string.format('"%s": "%s"', k, M.json_escape(job[k]))
  end
  parts[#parts + 1] = string.format('"start": %d', job.start)
  parts[#parts + 1] = string.format('"end": %d', job["end"])
  return M.write_all(job_path, "{" .. table.concat(parts, ", ") .. "}")
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
  return ms:AddClipMattesToMediaPool(ctx.mpi, { matte_path }) and true or false
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
  local start = tonumber(attrs.COMPN_RenderStart) or 0
  -- image source f  →  image de comp  f - in_frame + RenderStart
  return comp, math.floor(start - ctx.in_frame)
end

function M.make_setting(root, csv, mode, smooth, offset, out_path)
  local args = {
    M.python_exe(root), M.join(root, "fusion_export.py"), csv,
    "--mode", mode, "--smooth=" .. M.fmt_num(smooth),
    "--frame-offset=" .. M.fmt_num(offset), "-o", out_path,
  }
  local q = {}
  for i, v in ipairs(args) do q[i] = M.quote(v) end
  local cmd = table.concat(q, " ")
  if M.IS_WIN then cmd = '"' .. cmd .. '"' end  -- règle de guillemets de cmd /c
  os.execute(cmd)
  return M.read_all(out_path)
end

function M.paste_into_comp(comp, setting_text, insert_before_output)
  local data = bmd.readstring(setting_text)
  if not data then return false, "Fichier .setting illisible." end
  comp:Lock()
  comp:StartUndo("TAPNext++")
  local ok = comp:Paste(data)
  local msg = "Nœud ajouté dans la comp Fusion."
  if ok and insert_before_output then
    local sel = comp:GetToolList(true, "Transform") or {}
    local tool = sel[1]
    local mo = (comp:GetToolList(false, "MediaOut") or {})[1]
    if tool and mo then
      local src = mo.Input and mo.Input:GetConnectedOutput()
      if src then
        tool:ConnectInput("Input", src:GetTool())
      else
        local mi = (comp:GetToolList(false, "MediaIn") or {})[1]
        if mi then tool:ConnectInput("Input", mi) end
      end
      mo:ConnectInput("Input", tool)
      msg = "Nœud de stabilisation inséré avant " .. (mo.Name or "MediaOut") .. "."
    end
  end
  comp:EndUndo(true)
  comp:Unlock()
  return ok and true or false, msg
end

-- ------------------------------------------------------------- interface
local FUSION = { "none", "stabilize", "matchmove" }

function M.run_ui(resolve, fu)
  local ctx, err = M.get_context(resolve)
  local ui = fu.UIManager
  local disp = bmd.UIDispatcher(ui)

  if not ctx then
    local w = disp:AddWindow({ ID = "TAPErr", WindowTitle = "TAPNext++", Geometry = { 300, 300, 420, 140 } },
      ui:VGroup { ui:Label { Text = err, WordWrap = true }, ui:Button { ID = "Ok", Text = "OK" } })
    function w.On.Ok.Clicked() disp:ExitLoop() end
    function w.On.TAPErr.Close() disp:ExitLoop() end
    w:Show(); disp:RunLoop(); w:Hide()
    return
  end

  local root = TAPNEXT_ROOT
  local win = disp:AddWindow({
    ID = "TAPWin", WindowTitle = "TAPNext++ Tracker", Geometry = { 240, 160, 520, 380 },
  }, ui:VGroup {
    ui:Label { Weight = 0, WordWrap = true, Text = "<b>Clip :</b> " .. ctx.name ..
      "<br><small>" .. ctx.path .. "</small><br><small>Images source " .. ctx.in_frame ..
      " → " .. ctx.out_frame .. " (partie utilisée dans la timeline)</small>" },
    ui:HGroup { Weight = 0,
      ui:Label { Text = "Dossier de l'outil", MinimumSize = { 120, 20 } },
      ui:LineEdit { ID = "Root", Text = M.root_is_valid(root) and root or "" } },
    ui:CheckBox { ID = "Attach", Weight = 0, Checked = true,
      Text = "Attacher la matte au clip (page Color → Add Matte)" },
    ui:HGroup { Weight = 0,
      ui:Label { Text = "Nœud Fusion", MinimumSize = { 120, 20 } }, ui:ComboBox { ID = "Fusion" } },
    ui:HGroup { Weight = 0,
      ui:Label { Text = "Lissage stabilisation", MinimumSize = { 120, 20 } },
      ui:SpinBox { ID = "Smooth", Minimum = 0, Maximum = 200, Value = 0 } },
    ui:Button { ID = "Open", Weight = 0, Text = "Ouvrir dans TAPNext Studio" },
    ui:Label { ID = "Status", Weight = 1, WordWrap = true, Alignment = { AlignTop = true },
      Text = "Studio s'ouvre sur ce clip : entourez le sujet, lancez le suivi, puis " ..
        "« Exporter et envoyer à Resolve ». Les résultats arriveront ici automatiquement." },
    ui:HGroup { Weight = 0, ui:HGap(0, 1), ui:Button { ID = "Close", Text = "Fermer" } },
  })
  local itm = win:GetItems()
  for _, m in ipairs({ "Aucun", "Stabiliser (inséré dans la comp)", "Match-move (nœud ajouté)" }) do
    itm.Fusion:AddItem(m)
  end

  local state = { waiting = false }
  local timer = ui:Timer { ID = "Poll", Interval = 1000 }
  local function set_status(s) itm.Status.Text = s end

  local function import_results(done_text)
    local opts = state.opts
    local report = { "Résultats reçus de TAPNext Studio." }
    local matte = M.json_field(done_text, "matte")
    if itm.Attach.Checked and matte and matte ~= "" then
      if M.attach_matte(resolve, ctx, matte) then
        report[#report + 1] = "✔ Matte attachée au clip : page Color → clic droit → Add Matte."
      else
        report[#report + 1] = "Matte non attachée automatiquement ; importez : " .. matte
      end
    end
    local mode = FUSION[itm.Fusion.CurrentIndex + 1] or "none"
    local csv = M.json_field(done_text, "csv")
    if mode ~= "none" and csv then
      local comp, offset = M.get_comp(ctx)
      if comp then
        local setting = M.join(opts.out_dir, opts.name .. "_fusion_" .. mode .. ".setting")
        local text = M.make_setting(opts.root, csv, mode, itm.Smooth.Value, offset, setting)
        if text then
          local ok, msg = M.paste_into_comp(comp, text, mode == "stabilize")
          report[#report + 1] = ok and ("✔ " .. msg) or ("Collage Fusion impossible ; fichier : " .. setting)
        else
          report[#report + 1] = "Génération du nœud Fusion impossible."
        end
      else
        report[#report + 1] = "Comp Fusion introuvable pour ce clip."
      end
    end
    report[#report + 1] = "Fichiers : " .. opts.out_dir
    set_status(table.concat(report, "\n"))
  end

  function disp.On.Poll.Timeout()
    if not state.waiting then timer:Stop() return end
    local text = M.read_all(state.done)
    if not text or text == "" then return end
    timer:Stop()
    state.waiting = false
    itm.Open.Enabled = true
    if M.json_field(text, "status") == "ok" then
      import_results(text)
    else
      set_status("TAPNext Studio a été fermé sans export.")
    end
  end

  function win.On.Open.Clicked()
    local r = itm.Root.Text
    if not M.root_is_valid(r) then
      set_status("Dossier de l'outil invalide : indiquez le dossier qui contient tap_studio.py.")
      return
    end
    if not M.exists(M.python_exe(r)) then
      set_status("Environnement Python absent : lancez d'abord l'installateur dans " .. r)
      return
    end
    if M.IS_WIN and not M.is_ascii(r) then
      set_status("Le dossier de l'outil contient des caractères accentués : déplacez-le " ..
        "(ex. C:\\TAPNext) puis relancez l'installateur.")
      return
    end
    local out_dir = M.join(M.dirname(ctx.path), "TAPNext")
    if (M.IS_WIN and not M.is_ascii(out_dir)) or not M.dir_writable(out_dir) then
      out_dir = M.join(r, "output")
      M.mkdir(out_dir)
    end
    local base = M.basename_noext(ctx.path)
    if M.IS_WIN and not M.is_ascii(base) then base = "clip" end
    local name = base .. "_" .. ctx.in_frame .. "-" .. ctx.out_frame
    local opts = { root = r, out_dir = out_dir, name = name }
    local jobs = M.join(r, "jobs")
    M.mkdir(jobs)
    local job = M.join(jobs, "resolve_job.json")
    local done = M.join(jobs, "resolve_done.json")
    os.remove(done)
    M.write_job(job, { video = ctx.path, out_dir = out_dir, name = name, done = done,
      start = ctx.in_frame, ["end"] = ctx.out_frame })
    os.execute(M.studio_cmd(r, job))
    state.waiting, state.done, state.opts = true, done, opts
    itm.Open.Enabled = false
    set_status("TAPNext Studio est ouvert. En attente de l'export…\n(Gardez cette fenêtre ouverte.)")
    timer:Start()
  end

  local function close()
    timer:Stop()
    disp:ExitLoop()
  end
  win.On.Close.Clicked = close
  win.On.TAPWin.Close = close

  win:Show()
  disp:RunLoop()
  win:Hide()
end

-- ------------------------------------------------------------- point d'entrée
if not TAPNEXT_TEST then
  local res = resolve or (Resolve and Resolve())
  local fusion_obj = fu or fusion or (res and res:Fusion())
  if not res or not fusion_obj then
    print("TAPNext++ : ce script doit être lancé depuis DaVinci Resolve (Workspace > Scripts).")
  else
    M.run_ui(res, fusion_obj)
  end
end

return M
