--[[
TAPNext++ Tracker — intégration DaVinci Resolve
================================================

Script Resolve (Lua natif, aucune dépendance côté Resolve) qui apparaît dans
  Workspace > Scripts > TAPNext_Tracker
sur toutes les pages (dossier « Scripts/Utility »).

Ce qu'il fait, sur le clip situé sous la tête de lecture de la timeline :
  1. affiche une fenêtre de réglages (points, blobs, mouvement, modèle…) ;
  2. lance le moteur TAPNext++ (tap_resolve_tool.py, installé par
     INSTALLER_Windows.bat) en arrière-plan, uniquement sur la partie du clip
     utilisée dans la timeline, et affiche la progression ;
  3. à la fin, automatiquement :
       • attache la matte N&B au clip (Media Pool → clip matte), prête pour
         « Add Matte » dans la page Color ;
       • optionnel : ajoute le nœud Transform (stabilisation ou match-move)
         dans la composition Fusion du clip, avec les bonnes images-clés.

Installé automatiquement par INSTALLER_Windows.bat / install.sh (le chemin du
dossier de l'outil est inscrit ci-dessous).
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

-- ----------------------------------------------------- ligne de commande
--[[ opts : table de réglages (voir collect_options) ; renvoie la liste
     d'arguments passés à tap_resolve_tool.py. ]]
function M.build_args(root, opts)
  local a = {
    M.python_exe(root), "-u", M.join(root, "tap_resolve_tool.py"), opts.video,
    "--output-dir", opts.out_dir, "--name", opts.name,
    "--start-frame=" .. M.fmt_num(opts.start_frame),
    "--end-frame=" .. M.fmt_num(opts.end_frame),
    "--radius=" .. M.fmt_num(opts.radius),
    "--merge=" .. M.fmt_num(opts.merge),
    "--threshold=" .. M.fmt_num(opts.threshold),
    "--motion-mode", opts.motion_mode,
    "--motion-sensitivity=" .. M.fmt_num(opts.motion_sensitivity),
    "--fade-in=" .. M.fmt_num(opts.fade_in),
    "--fade-out=" .. M.fmt_num(opts.fade_out),
    "--input-res", tostring(opts.input_res),
    "--codec", "prores",
    -- Le nœud Fusion est généré après coup (décalage d'image exact de la comp).
    "--fusion-mode", "none",
  }
  if opts.pick then
    a[#a + 1] = "--pick"
  else
    a[#a + 1] = "--grid=" .. M.fmt_num(opts.grid)
  end
  if opts.fp16 then a[#a + 1] = "--fp16-weights" end
  if opts.preview then a[#a + 1] = "--preview" end
  return a
end

function M.outputs(opts)
  local base = M.join(opts.out_dir, opts.name)
  return {
    matte = base .. "_matte.mov",
    csv = base .. "_tracks.csv",
    log = base .. "_log.txt",
    done = base .. "_done.txt",
    script = base .. (M.IS_WIN and "_run.bat" or "_run.sh"),
  }
end

--[[ Écrit le script de lancement (journal + code de sortie dans *_done.txt). ]]
function M.write_launcher(args, out)
  local q = {}
  for i, v in ipairs(args) do q[i] = M.quote(v, true) end
  local cmd = table.concat(q, " ")
  local body
  if M.IS_WIN then
    body = table.concat({
      "@echo off",
      "title TAPNext++ (fermer cette fenetre pour annuler)",
      "chcp 65001 >nul",
      "set PYTHONIOENCODING=utf-8",
      "set PYTHONUTF8=1",
      'del ' .. M.quote(out.done, true) .. ' >nul 2>&1',
      cmd .. " > " .. M.quote(out.log, true) .. " 2>&1",
      ">" .. M.quote(out.done, true) .. " echo %ERRORLEVEL%",
      "",
    }, "\r\n")
  else
    body = table.concat({
      "#!/bin/sh",
      "export PYTHONIOENCODING=utf-8 PYTHONUTF8=1",
      "rm -f " .. M.quote(out.done),
      cmd .. " > " .. M.quote(out.log) .. " 2>&1",
      "echo $? > " .. M.quote(out.done),
      "",
    }, "\n")
  end
  return M.write_all(out.script, body)
end

function M.launch(out)
  M.write_all(out.log, "")
  os.remove(out.done)
  if M.IS_WIN then
    -- Fenêtre réduite : la fermer annule le traitement.
    return os.execute('start "TAPNext++" /min ' .. M.quote(out.script))
  end
  return os.execute("sh " .. M.quote(out.script) .. " >/dev/null 2>&1 &")
end

--[[ Lit le journal : renvoie (ligne d'état courante, N dernières lignes). ]]
function M.parse_log(text, n)
  n = n or 12
  local status, lines = "", {}
  if not text or text == "" then return status, "" end
  for seg in text:gmatch("[^\r\n]+") do
    if seg:find("%d+%%|") then
      status = seg                       -- barre de progression tqdm
    elseif seg:match("%S") then
      lines[#lines + 1] = seg
      status = seg
    end
  end
  local first = math.max(1, #lines - n + 1)
  local tail = {}
  for i = first, #lines do tail[#tail + 1] = lines[i] end
  return status, table.concat(tail, "\n")
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
local MOTION = { "none", "scale", "stretch", "both" }
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
  local function row(label, widget)
    return ui:HGroup { Weight = 0, ui:Label { Text = label, MinimumSize = { 190, 20 } }, widget }
  end

  local win = disp:AddWindow({
    ID = "TAPWin", WindowTitle = "TAPNext++ Tracker", Geometry = { 200, 120, 540, 760 },
  }, ui:VGroup {
    ui:Label { Weight = 0, Text = "<b>Clip :</b> " .. ctx.name ..
      "<br><small>" .. ctx.path .. "</small>", WordWrap = true },
    row("Dossier de l'outil TAPNext", ui:LineEdit { ID = "Root", Text = M.root_is_valid(root) and root or "" }),
    ui:Label { Weight = 0, Text = "<b>Points</b>" },
    row("Initialisation", ui:ComboBox { ID = "PointsMode" }),
    row("Grille N×N", ui:SpinBox { ID = "Grid", Minimum = 1, Maximum = 64, Value = 10 }),
    row("Image source début", ui:SpinBox { ID = "Start", Minimum = 0, Maximum = 10000000, Value = ctx.in_frame }),
    row("Image source fin", ui:SpinBox { ID = "End", Minimum = 0, Maximum = 10000000, Value = ctx.out_frame }),
    ui:Label { Weight = 0, Text = "<b>Matte</b>" },
    row("Rayon des blobs (px)", ui:DoubleSpinBox { ID = "Radius", Minimum = 1, Maximum = 2000, Value = 40, Decimals = 1 }),
    row("Fusion des blobs (× rayon)", ui:DoubleSpinBox { ID = "Merge", Minimum = 0.05, Maximum = 5, Value = 0.6, SingleStep = 0.05, Decimals = 2 }),
    row("Seuil metaball", ui:DoubleSpinBox { ID = "Threshold", Minimum = 0.05, Maximum = 0.95, Value = 0.5, SingleStep = 0.05, Decimals = 2 }),
    row("Réaction au mouvement", ui:ComboBox { ID = "Motion" }),
    row("Sensibilité au mouvement", ui:DoubleSpinBox { ID = "Sens", Minimum = 0, Maximum = 5, Value = 0.05, SingleStep = 0.01, Decimals = 3 }),
    row("Fondu apparition (images)", ui:SpinBox { ID = "FadeIn", Minimum = 0, Maximum = 200, Value = 6 }),
    row("Fondu disparition (images)", ui:SpinBox { ID = "FadeOut", Minimum = 0, Maximum = 200, Value = 8 }),
    ui:Label { Weight = 0, Text = "<b>Modèle et résultats</b>" },
    row("Résolution interne", ui:ComboBox { ID = "Res" }),
    ui:CheckBox { ID = "FP16", Weight = 0, Text = "Poids FP16 (moins de VRAM)", Checked = false },
    ui:CheckBox { ID = "Attach", Weight = 0, Text = "Attacher la matte au clip (page Color)", Checked = true },
    row("Nœud Fusion", ui:ComboBox { ID = "Fusion" }),
    row("Lissage stabilisation (0 = verrouillé)", ui:SpinBox { ID = "Smooth", Minimum = 0, Maximum = 200, Value = 0 }),
    ui:CheckBox { ID = "Preview", Weight = 0, Text = "Vidéo de contrôle (_preview.mp4)", Checked = false },
    ui:HGroup { Weight = 0,
      ui:Button { ID = "Run", Text = "Lancer le tracking" },
      ui:Button { ID = "Close", Text = "Fermer" },
    },
    ui:Label { ID = "Status", Weight = 0, Text = "Prêt.", WordWrap = true },
    ui:TextEdit { ID = "Log", ReadOnly = true, Weight = 1 },
  })
  local itm = win:GetItems()
  itm.PointsMode:AddItem("Grille automatique")
  itm.PointsMode:AddItem("Cliquer les points sur l'image")
  for _, m in ipairs({ "Aucune", "Agrandir (scale)", "Étirer (stretch)", "Les deux (both)" }) do itm.Motion:AddItem(m) end
  itm.Res:AddItem("512 (précis)")
  itm.Res:AddItem("256 (rapide)")
  for _, m in ipairs({ "Aucun", "Stabiliser (inséré dans la comp)", "Match-move (nœud ajouté)" }) do itm.Fusion:AddItem(m) end

  local state = { running = false }
  local timer = ui:Timer { ID = "Poll", Interval = 1000 }

  local function set_status(s) itm.Status.Text = s end

  local function finish()
    local out, opts = state.out, state.opts
    local code = tonumber((M.read_all(out.done) or ""):match("%-?%d+")) or -1
    local _, tail = M.parse_log(M.read_all(out.log), 40)
    itm.Log.PlainText = tail
    state.running = false
    itm.Run.Enabled = true
    if code ~= 0 then
      set_status("Échec du traitement (code " .. code .. "). Journal : " .. out.log)
      return
    end
    local report = { "Tracking terminé." }
    if opts.attach then
      if M.attach_matte(resolve, ctx, out.matte) then
        report[#report + 1] = "Matte attachée au clip : page Color → clic droit → Add Matte."
      else
        report[#report + 1] = "Impossible d'attacher la matte automatiquement ; importez " .. out.matte
      end
    end
    if opts.fusion_mode ~= "none" then
      local comp, offset = M.get_comp(ctx)
      if comp then
        local setting = M.join(opts.out_dir, opts.name .. "_fusion_" .. opts.fusion_mode .. ".setting")
        local text = M.make_setting(opts.root, out.csv, opts.fusion_mode, opts.smooth, offset, setting)
        if text then
          local ok, msg = M.paste_into_comp(comp, text, opts.fusion_mode == "stabilize")
          report[#report + 1] = ok and msg or ("Collage Fusion impossible ; fichier : " .. setting)
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
    if not state.running then timer:Stop() return end
    local status, tail = M.parse_log(M.read_all(state.out.log), 12)
    if status ~= "" then set_status(status) end
    itm.Log.PlainText = tail
    if M.exists(state.out.done) then
      timer:Stop()
      finish()
    end
  end

  function win.On.Run.Clicked()
    if state.running then return end
    local r = itm.Root.Text
    if not M.root_is_valid(r) then
      set_status("Dossier de l'outil invalide : indiquez le dossier qui contient tap_resolve_tool.py.")
      return
    end
    if not M.exists(M.python_exe(r)) then
      set_status("Environnement Python absent : lancez d'abord l'installateur dans " .. r)
      return
    end
    local out_dir = M.join(M.dirname(ctx.path), "TAPNext")
    if not M.dir_writable(out_dir) then out_dir = M.join(r, "output") ; M.mkdir(out_dir) end
    local s, e = itm.Start.Value, itm.End.Value
    if e < s then s, e = e, s end
    local opts = {
      root = r, video = ctx.path, out_dir = out_dir,
      name = M.basename_noext(ctx.path) .. "_" .. s .. "-" .. e,
      start_frame = s, end_frame = e,
      pick = itm.PointsMode.CurrentIndex == 1, grid = itm.Grid.Value,
      radius = itm.Radius.Value, merge = itm.Merge.Value, threshold = itm.Threshold.Value,
      motion_mode = MOTION[itm.Motion.CurrentIndex + 1] or "none",
      motion_sensitivity = itm.Sens.Value, fade_in = itm.FadeIn.Value, fade_out = itm.FadeOut.Value,
      input_res = itm.Res.CurrentIndex == 1 and 256 or 512, fp16 = itm.FP16.Checked,
      attach = itm.Attach.Checked, fusion_mode = FUSION[itm.Fusion.CurrentIndex + 1] or "none",
      smooth = itm.Smooth.Value, preview = itm.Preview.Checked,
    }
    local out = M.outputs(opts)
    local ok, werr = M.write_launcher(M.build_args(r, opts), out)
    if not ok then set_status("Écriture impossible : " .. tostring(werr)) return end
    M.launch(out)
    state.running, state.out, state.opts = true, out, opts
    itm.Run.Enabled = false
    set_status(opts.pick and "Cliquez les points dans la fenêtre qui s'ouvre, puis Entrée…"
      or "Chargement du modèle…")
    timer:Start()
  end

  local function close()
    if state.running then
      set_status("Traitement en cours : il continuera en arrière-plan, mais l'import automatique " ..
        "n'aura pas lieu. Fermez à nouveau pour confirmer.")
      state.running = false
      return
    end
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
