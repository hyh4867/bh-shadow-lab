# 자료 (DATA)

원자료는 git에 넣지 않는다. 아래 방법으로 받아 `data/raw/`에 둔다(`.gitignore`에 포함).

## 자료 정보

- 이름: First M87 EHT Results: Calibrated Data (EHT 자료 코드 2019-D01-01)
- 저자: The Event Horizon Telescope Collaboration et al.
- DOI: [10.25739/g85n-f134](https://doi.org/10.25739/g85n-f134) (CyVerse Data Commons)
- 라이선스: 자료 파일은 **ODC-PDDL 1.0**(Open Data Commons Public Domain Dedication and License). 함께 들어 있는 스크립트(`run.sh`, `convert_stokesI.py`, `dump_*.py`)는 GPLv3(릴리스의 `LICENSE.txt`). 이 저장소는 스크립트를 쓰지도 복사하지도 않는다.
- 내용: 2017-04-05, 06, 10, 11(day of year 095, 096, 100, 101) × lo, hi 밴드의 M87 Stokes I visibility. EHT-HOPS 파이프라인의 network-calibrated 자료이고, 10초 시간 평균, 32 IF 주파수 평균이다. RR과 LL에는 같은 Stokes I 값이 들어 있고 각 오차는 Stokes I 오차의 √2배다(릴리스 README). 관측 중심 주파수는 lo 227.07 GHz, hi 229.07 GHz다(uvfits 헤더 CRVAL4).

### 인용 문구

자료 페이지의 권장 인용:

> The Event Horizon Telescope Collaboration 2019. First M87 EHT Results: Calibrated Data. CyVerse Data Commons. DOI 10.25739/g85n-f134

자료의 주 참고문헌: The Event Horizon Telescope Collaboration et al. 2019c, ApJL 875, L3 (M87 Paper III).

## 받는 방법

DOI가 가리키는 CyVerse 익명 다운로드 주소(`https://data.cyverse.org/dav-anon/iplant/home/shared/commons_repo/curated/EHTC_FirstM87Results_Apr2019/`)는 2026-11-14에 Cloudflare 사람 인증 화면(`unblockme.cyverse.org`)으로 넘어가서 스크립트로 받을 수 없었다. 그래서 EHT 공식 GitHub에 올라온 같은 릴리스에서 받았다. 두 곳은 파일 목록, README, INVENTORY가 같다.

```bash
cd projects/bh-shadow-lab
mkdir -p data/raw/dl && cd data/raw/dl
for f in EHTC_FirstM87Results_Apr2019_uvfits.tgz EHTC_FirstM87Results_Apr2019_csv.tgz; do
  curl -sSL -o $f "https://raw.githubusercontent.com/eventhorizontelescope/2019-D01-01/master/$f"
done
shasum -a 256 *.tgz          # 아래 표와 비교
mkdir -p ../x && tar -xzf EHTC_FirstM87Results_Apr2019_uvfits.tgz -C ../x && tar -xzf EHTC_FirstM87Results_Apr2019_csv.tgz -C ../x
cd .. && mkdir -p uvfits csv
cp x/EHTC_FirstM87Results_Apr2019/uvfits/*.uvfits uvfits/
cp x/EHTC_FirstM87Results_Apr2019/csv/*.csv csv/
rm -rf x
```

CyVerse에서 브라우저로 받은 경우에도 같은 SHA-256이어야 한다. CyVerse 판과의 해시 비교는 아직 하지 못했다(미확인).

## SHA-256 (2026-11-14에 받은 파일)

| 파일 | SHA-256 |
|---|---|
| EHTC_FirstM87Results_Apr2019_uvfits.tgz | f36f77fe4c0cf772625dcf424ed34ba3d1b5766475c64ed90319b0000f1ed463 |
| EHTC_FirstM87Results_Apr2019_csv.tgz | 8fa8964ad5b1c80dd78bc3e023f849bb33312b2d8af95d6718439f529f361911 |
| SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits | a11a874bdbf205089192b5270abc5894ad85e4ab54f0f677e3280c5639ea3e0d |
| SR1_M87_2017_095_lo_hops_netcal_StokesI.uvfits | 97492ee1146b05e12b1783e6c9a8253a0f50d789835286ffcc3ed5d248d51a8e |
| SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits | 5bc42247ccd01b58460d95e8c17f91e337c79ffe99a0718aacf0a91c77ad24ec |
| SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits | f3394b2c272ba92980c21e3522987344f9175a78c333648a36458c917216c365 |
| SR1_M87_2017_100_hi_hops_netcal_StokesI.uvfits | 76104537313cd6879c497901f313bad667f4608c48df961e67ac5615d84b2c00 |
| SR1_M87_2017_100_lo_hops_netcal_StokesI.uvfits | ee90ca5aec0bd8ed27f5ea13c941ae3f24c188d5c8a9aa87ab4b5606f57334be |
| SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits | 618c3019f60e88268980267a9db68f638b379a85842d83c06003d28550d191f5 |
| SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits | 697af2bb3bbf732115108ffefacd3e59e307f38fe685c3c4579146b0bd661298 |

추가 확인: uvfits 8개의 git blob SHA-1(`git hash-object`)이 GitHub API의 파일 목록(`repos/eventhorizontelescope/2019-D01-01/contents/uvfits`)과 모두 같았다.

csv 파일(reader 확인용):

| 파일 | SHA-256 |
|---|---|
| SR1_M87_2017_095_hi_hops_netcal_StokesI.csv | 1e7c2a63e79b1722dcc5d7fe73d8193309adec20c4e660b0659169ee66c63541 |
| SR1_M87_2017_095_lo_hops_netcal_StokesI.csv | e11231fbb09a29762dbb318a733be329a1dfea2ba2cad4e13a6208766b0134a8 |
| SR1_M87_2017_096_hi_hops_netcal_StokesI.csv | 769e5ab12c47f3f4ad5da99e7ed44594111717e831c5aa5e4a5c366cfc0bd6b6 |
| SR1_M87_2017_096_lo_hops_netcal_StokesI.csv | fd4d1b39ab2e561f502d70c93667d67a525aa9ac4af432ceed01669e039911c5 |
| SR1_M87_2017_100_hi_hops_netcal_StokesI.csv | 3e6f034e12212a862632e6242b26335e4b48c68cfa87647632834cd43714a353 |
| SR1_M87_2017_100_lo_hops_netcal_StokesI.csv | 1dc51e3e31a840ec34de5b5ff4d8a6af0cbb73cd8c6db85e917366ed3ec7924a |
| SR1_M87_2017_101_hi_hops_netcal_StokesI.csv | 7710f9a5c830ded0749e4f52ced75438435faf8bfb0b6e11b9e3adec25489b92 |
| SR1_M87_2017_101_lo_hops_netcal_StokesI.csv | 6411680cdd4a0b6a651b56c8ada9069ae49659935877ff7e03acb9b173a80bb2 |

## 파일 구조 메모 (읽은 뒤 확인한 것)

- 주 HDU는 random groups다. 매개변수는 UU, VV, WW(초 단위. 주파수를 곱하면 파장 단위가 된다), BASELINE(256·ant1 + ant2), DATE(두 열을 합하면 JD), INTTIM이다. 자료 축은 (complex 3, stokes 4: RR, LL, RL, LR)이다.
- AIPS AN 표에는 국 8개가 있다(AA, AP, AZ, JC, LM, PV, SM, SR). SR은 SM과 좌표가 같고 visibility에는 나오지 않는다.
- **scan 표(NX)가 없다.** 사전 등록대로 시각 간격 60초 이상을 scan 경계로 쓴다. 실제 간격은 scan 안에서 10초, scan 사이에서 최소 190초라 경계가 분명하다. scan 개수는 4/5 18, 4/6 25, 4/10 7, 4/11 22개다.
- RR = LL을 8개 파일 모두에서 확인했다. 가중치 0이나 NaN인 점은 없었다.
- 푸리에 부호: 이 파일은 eht-imaging(+2πi 규약)으로 썼다. 우리는 읽을 때 켤레를 취한다(THEORY.md 2절).
- 독립 확인: 우리가 읽은 시각, u, v, 진폭, 위상, 오차를 릴리스의 csv(EHT가 만든 파생 파일)와 비교했다. 최대 상대 차이는 약 1e-5로 csv의 반올림 수준이다(`results/stage1_reader_check.json`).

## 2018 자료 (5단계 추가 분석)

- 이름: 2018 M87 EHT Results: Calibrated Data (EHT 자료 코드 2024-D01-01)
- 저자: The Event Horizon Telescope Collaboration et al.
- 주 참고문헌: The Event Horizon Telescope Collaboration et al. 2024, A&A 681, A79, DOI [10.1051/0004-6361/202347932](https://doi.org/10.1051/0004-6361/202347932)
- 받은 곳: EHT 공식 GitHub [eventhorizontelescope/2024-D01-01](https://github.com/eventhorizontelescope/2024-D01-01). 받은 시점의 저장소 commit은 fda41da417419fd9b54505ea3fa70a8cf6fb60ec(2024-03-11)이다. EHT 자료 포털(eventhorizontelescope.org/for-astronomers/data)이 이 저장소를 가리킨다.
- 라이선스: 자료 파일은 **ODC-PDDL 1.0**이고, 스크립트는 GPLv3다(저장소의 `LICENSE.txt`). 스크립트는 쓰지 않았다.
- 내용: 2018-04-21, 22, 25(day of year 111, 112, 115) × 밴드 b1–b4(213.1, 215.1, 227.1, 229.1 GHz). 10초 평균이고 32 IF를 평균했으며 Stokes I다(RR = LL, 오차 √2배). EHT-HOPS와 CASA rPICARD 두 파이프라인 판이 있고, 우리는 2017과 같은 HOPS판만 쓴다. README는 셋째 날을 "April 27th"라고 적었지만 파일 이름의 115는 4/25이고, 논문도 4/25다.
- 국 코드는 2017과 다르다. AX = APEX, MM = JCMT, SW = SMA, MG = SMT이고 GL(Greenland Telescope)이 새로 들어갔다. intra-site 기선은 AA–AX와 MM–SW다.

### 인용 문구 (2018)

> The Event Horizon Telescope Collaboration et al. 2024, "The persistent shadow of the supermassive black hole of M 87. I. Observations, calibration, imaging, and analysis", A&A 681, A79. Data: 2018 M87 EHT Results: Calibrated Data (2024-D01-01).

### 받는 방법 (2018)

```bash
cd projects/bh-shadow-lab
mkdir -p data/raw/uvfits2018 data/raw/csv2018
for d in 111 112 115; do for b in b1 b2 b3 b4; do
  f=L2V1_M87_2018_${d}_${b}_hops_netcal_10s_StokesI.uvfits
  curl -sSfL -o data/raw/uvfits2018/$f "https://raw.githubusercontent.com/eventhorizontelescope/2024-D01-01/HEAD/uvfits/$f"
done; done
f=L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.csv   # reader 확인용
curl -sSfL -o data/raw/csv2018/$f "https://raw.githubusercontent.com/eventhorizontelescope/2024-D01-01/HEAD/csv/$f"
shasum -a 256 data/raw/uvfits2018/*.uvfits data/raw/csv2018/*.csv   # 아래 표와 비교
```

### SHA-256 (2026-11-14에 받은 파일)

| 파일 | 크기 (byte) | SHA-256 |
|---|---|---|
| L2V1_M87_2018_111_b1_hops_netcal_10s_StokesI.uvfits | 452160 | 50917838a075c98b800da71f5b604c75e50c7d97c11a4da73afa261f67432e32 |
| L2V1_M87_2018_111_b2_hops_netcal_10s_StokesI.uvfits | 394560 | 8441ac42c9600302ee34719c334b837712da78ef1962d33bcf7d316416a23d23 |
| L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits | 642240 | 97e14f6f53bf7101acea7a107452a8f587b49ddccbd788d06f3b44dbdf5f694a |
| L2V1_M87_2018_111_b4_hops_netcal_10s_StokesI.uvfits | 699840 | 630a679d1c75cc85c4849da61d20a7422aa0530bb1b8e3ee3b599ed7ef2bffab |
| L2V1_M87_2018_112_b1_hops_netcal_10s_StokesI.uvfits | 97920 | 9a97cce232e2e5887c93e490e2a0705568bdf122781b8a3126c82675c46db756 |
| L2V1_M87_2018_112_b2_hops_netcal_10s_StokesI.uvfits | 100800 | 29763f752c88ec30388538bed1a405bfb6b7f3d7ccdb601ca3a84d0dbf6182a5 |
| L2V1_M87_2018_112_b3_hops_netcal_10s_StokesI.uvfits | 152640 | 24906905c27ac6d488b6b5e1caeca7968eb318e2dd79f18296c1c7a816f4a30e |
| L2V1_M87_2018_112_b4_hops_netcal_10s_StokesI.uvfits | 161280 | 63fb21bd1cd80671a34e635616756bea7e4b5d9a1ae9ee208a783b575f2d8cb0 |
| L2V1_M87_2018_115_b1_hops_netcal_10s_StokesI.uvfits | 285120 | 619f08697826e04114fad33ce497f312878d2c39b0c5a51c13751fdabf39340f |
| L2V1_M87_2018_115_b2_hops_netcal_10s_StokesI.uvfits | 288000 | 82ea9c8d81cc3b457f16f97443cbcacdc335e4e4fb6b4dec84469b29c68a56f7 |
| L2V1_M87_2018_115_b3_hops_netcal_10s_StokesI.uvfits | 460800 | 8c290e67fcbb2aa943520aaa90aad076c5ece5e6ab1ac4674507c1b1d21e5f0e |
| L2V1_M87_2018_115_b4_hops_netcal_10s_StokesI.uvfits | 489600 | f218f2bd9a546a2d294e88d617128ac91b36a17b23ee183bb764c0b8f6d22532 |
| L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.csv | | 79bdbcb5f8124323bff733095e6fa998071b8a5bce4edbbba26f4f83eb652187 |

- uvfits 12개의 git blob SHA-1은 GitHub API 파일 목록(`repos/eventhorizontelescope/2024-D01-01/git/trees/HEAD`)과 모두 같았다.
- 파일 구조는 2017과 같다(eht-imaging으로 쓴 random groups이고 헤더도 같은 형식이다). 그래서 2017처럼 읽을 때 켤레를 취한다.
- 4/21 파일에는 AIPS NX 표가 있지만 행이 0개이고, 나머지 파일에는 NX가 없다. 그래서 사전 등록 4절대로 시각 간격 60초 이상을 scan 경계로 쓴다.
