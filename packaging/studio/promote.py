"""Promote one exact, successful native build; never rebuild or overwrite a release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--commit', required=True)
parser.add_argument('--tag', required=True)
args = parser.parse_args()


def gh(*parts):
    return subprocess.check_output(['gh', *parts], text=True)


run = json.loads(gh('run', 'view', args.run, '--json', 'headSha,status,conclusion,workflowName'))
assert run == dict(headSha=args.commit, status='completed', conclusion='success',
                   workflowName='Studio desktop builds'), run
labels = ('windows-x64', 'linux-x64', 'macos-arm64', 'macos-x64')
cases = {'active_vector', 'passive_vector', 'active_das_line', 'passive_das_line',
         'active_das_array', 'passive_das_array', 'nonuniform_array', 'hvsr'}
incoming = Path('studio-promotion')
incoming.mkdir(exist_ok=False)
assets, checksums, reports = [], [], {}
for label in labels:
    for prefix in ('studio-', 'studio-qc-'):
        gh('run', 'download', args.run, '--name', prefix+label, '--dir', str(incoming/(prefix+label)))
    package_dir = incoming/('studio-'+label)
    qc = incoming/('studio-qc-'+label)
    report = json.loads((qc/'studio-smoke-output/self_test.json').read_text(encoding='utf-8'))
    manifest = json.loads((qc/'packaging/studio/build_manifest.json').read_text(encoding='utf-8'))
    assert manifest['git_commit'] == args.commit
    assert report['passed'] and set(report['cases']) == cases
    assert all(case['passed'] for case in report['cases'].values())
    assert report['legacy_active_default'] == 'passed' and report['peak_rss_bytes'] > 0
    lines = (package_dir/'SHA256SUMS.txt').read_text().strip().splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split()
    expected_name = 'WaveDispX-Studio-'+label+('.zip' if label == 'windows-x64' else '.tar.gz')
    assert name == expected_name
    path = package_dir/name
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    assets.append(str(path))
    checksums.append(digest+'  '+name)
    reports[label] = {'git_commit':args.commit, 'run_id':args.run, 'sha256':digest,
        'max_relative_L2':max(v for c in report['cases'].values() for v in c['relative_L2'].values()),
        'case_count':len(cases), 'passed':True}

summary = incoming/'Studio-platform-validation.json'
summary.write_text(json.dumps(reports, indent=2), encoding='utf-8')
sums = incoming/'SHA256SUMS.txt'
sums.write_text('\n'.join(checksums)+'\n', encoding='utf-8')
guide = incoming/'WaveDispX-Studio-guide-zh.zip'
with zipfile.ZipFile(guide, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in (Path('docs/studio'), Path('examples/custom_geometry')):
        for path in sorted(folder.rglob('*')):
            if path.is_file(): archive.write(path, path.as_posix())
    archive.write('packaging/studio/README_zh.md', 'INSTALL_zh.md')
    archive.write(summary, summary.name)
    for label in labels:
        qc = incoming/('studio-qc-'+label)
        for relative in ('studio-smoke-output/self_test.json',
                         'packaging/studio/build_manifest.json',
                         'studio-smoke-output/about.png',
                         'studio-smoke-output/active_vector_love.png'):
            archive.write(qc/relative, 'validation/'+label+'/'+Path(relative).name)
assets += [str(sums), str(summary), str(guide)]
notes = incoming/'RELEASE_zh.md'
notes.write_text('''# WaveDispX Studio 桌面试用版

综合正演工作台（不是 Classroom 教学版）：主动/被动源、节点/DAS、Geometry、真实炮集/NCF、频散能量与理论对照，以及可选 HVSR。CPU 运行；包内包含运行环境、原生求解器、中文字体和示例，无需另装 Python/Conda。

左上角可点击个人网站 https://runningxinliu.github.io/ ，联系 xinliu_geo@outlook.com。关于窗口感谢 Pan Lei 的 QEDispInv 仓库 https://github.com/pan3rock/QEDispInv ，并保留原程序与依赖来源。

## 下载与打开

- Windows 64位：windows-x64.zip，完整解压，双击 WaveDispX-Studio.exe。
- Apple 芯片 Mac：macos-arm64.tar.gz，解压打开 .app。
- Intel Mac：macos-x64.tar.gz，解压打开 .app。
- Linux 64位：linux-x64.tar.gz，完整解压运行 WaveDispX-Studio；构建/验收系统 Ubuntu 22.04。
- 保留整个目录，不要单独复制可执行文件。另下载 guide-zh.zip 查看中文操作说明、示例和验收记录；Source code 不是安装包。

当前为未签名、未公证试用包，系统可能显示安全提示；请确认下载来源和 SHA256，不要全局关闭系统安全保护。默认普通窗口，结果保存在用户可写目录；外部 CSV 会随保存结果留存快照。

## 验收边界

四个平台各自原生构建，在中文/空格路径运行实际 GUI，8 个固定案例及默认主动兼容流程通过，检查原始波形/NCF/频散/HVSR、保存后重读和中文字体。波形/能量/HVSR相对L2≤1e-5，不作增益或相位拟合。详细来源、指标及SHA256见附件。

这是平台移植回归，不是所有模型的精度或统计收敛认证。Love-DAS、逐源方向目录、远程任务提交和强制取消未实现。此版不改变物理核或自动切换理论求解器。保留全部上游声明；仓库尚无统一顶层许可证，上游许可证表述仍需进一步核对，不宣称完整许可审计完成。
'''+f'\n构建提交：`{args.commit}`；Actions：{args.run}。\n', encoding='utf-8')

# Refuse an existing published/draft tag: a retry needs explicit operator inspection.
probe = subprocess.run(['gh','release','view',args.tag], capture_output=True)
if probe.returncode == 0:
    raise RuntimeError('Release already exists; inspect it rather than overwrite assets')
gh('release','create',args.tag,'--target',args.commit,'--draft','--prerelease',
   '--title','WaveDispX Studio · 2026.10.28 RC1','--notes-file',str(notes))
gh('release','upload',args.tag,*assets)
gh('release','edit',args.tag,'--draft=false')
print(gh('release','view',args.tag,'--json','url','--jq','.url'))
