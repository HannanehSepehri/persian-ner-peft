# 🔍 Persian Named Entity Recognition with PEFT and MAD-X

تشخیص موجودیت‌های نامدار فارسی با استفاده از `XLM-RoBERTa-base` و مقایسه روش‌های **Full Fine-tuning، LoRA، Bottleneck Adapter** و **MAD-X Cross-lingual Zero-shot Transfer**

---

## 📌 درباره پروژه

این پروژه دو بخش اصلی دارد:

1. مقایسه سه روش آموزش برای NER فارسی روی دیتاست ARMAN:
   - Full Fine-tuning
   - LoRA
   - Bottleneck Adapter

2. پیاده‌سازی یک نسخه سبک و آموزشی از MAD-X برای انتقال بین‌زبانی NER:
   - آموزش Language Adapter فارسی
   - آموزش Language Adapter انگلیسی
   - آموزش NER Task Adapter فقط با داده برچسب‌دار انگلیسی
   - ارزیابی Zero-shot روی فارسی بدون استفاده از داده برچسب‌دار فارسی در آموزش Task Adapter

مدل پایه در هر دو بخش:

```text
xlm-roberta-base
```

محیط اجرا:

```text
Google Colab — Tesla T4 GPU
```

---

# بخش اول: مقایسه PEFT روی NER فارسی

## 📊 دیتاست ARMAN

منبع:

```text
hezarai/arman-ner
```

دیتاست ARMAN یکی از دیتاست‌های فارسی تشخیص موجودیت نامدار است. هر نمونه شامل کلمات جمله و برچسب NER مربوط به هر کلمه است.

تقسیم‌بندی نهایی داده‌ها:

| بخش | تعداد نمونه |
|---|---:|
| Train | 18,435 |
| Validation | 2,049 |
| Test | 2,561 |

مجموعه آموزشی با `seed=42` به بخش‌های Train و Validation تقسیم شد و مجموعه Test تا پایان آموزش بدون تغییر باقی ماند.

## 🏷️ برچسب‌های ARMAN

در این بخش از قالب BIO و در مجموع **۱۳ برچسب** استفاده شده است:

| ID | برچسب | معنی |
|---:|---|---|
| 0 | `O` | خارج از هر موجودیت |
| 1 | `B-pro` | شروع موجودیت محصول |
| 2 | `I-pro` | ادامه موجودیت محصول |
| 3 | `B-pers` | شروع موجودیت شخص |
| 4 | `I-pers` | ادامه موجودیت شخص |
| 5 | `B-org` | شروع موجودیت سازمان |
| 6 | `I-org` | ادامه موجودیت سازمان |
| 7 | `B-loc` | شروع موجودیت مکان |
| 8 | `I-loc` | ادامه موجودیت مکان |
| 9 | `B-fac` | شروع موجودیت تأسیسات |
| 10 | `I-fac` | ادامه موجودیت تأسیسات |
| 11 | `B-event` | شروع موجودیت رویداد |
| 12 | `I-event` | ادامه موجودیت رویداد |

شش نوع موجودیت موجود در دیتاست:

- `pers` — شخص، مانند «علی دایی»
- `org` — سازمان، مانند «سازمان ملل»
- `loc` — مکان، مانند «ایران»
- `fac` — تأسیسات یا بنا، مانند «برج میلاد»
- `pro` — محصول، مانند «آیفون»
- `event` — رویداد، مانند «جام جهانی»

در قالب BIO:

- `B` نشان‌دهنده شروع یک موجودیت است.
- `I` نشان‌دهنده ادامه همان موجودیت است.
- `O` نشان‌دهنده کلمه‌ای خارج از موجودیت است.

## 🔧 آماده‌سازی داده ARMAN

توکنایزر XLM-RoBERTa ممکن است یک کلمه فارسی را به چند زیرتوکن تقسیم کند.

برای هماهنگ‌کردن برچسب‌ها با زیرتوکن‌ها:

- اولین زیرتوکن، برچسب اصلی کلمه را دریافت کرد.
- زیرتوکن‌های بعدی مقدار `-100` گرفتند.
- توکن‌های ویژه نیز مقدار `-100` گرفتند.
- مقادیر `-100` در محاسبه Loss و معیارهای ارزیابی نادیده گرفته شدند.

حداکثر طول ورودی:

```text
256 tokens
```

## 🤖 نتایج Full Fine-tuning، LoRA و Adapter

| روش | Precision | Recall | F1-Score | Accuracy | درصد پارامتر قابل‌آموزش |
|---|---:|---:|---:|---:|---:|
| Full Fine-tuning | 0.9368 | 0.9650 | **0.9507** | 0.9945 | 100% |
| LoRA | 0.7483 | 0.8152 | 0.7803 | 0.9743 | **0.1098%** |
| Bottleneck Adapter | 0.8143 | 0.8644 | **0.8386** | 0.9820 | 0.6243% |

### تعداد پارامترهای قابل‌آموزش

| روش | پارامتر قابل‌آموزش |
|---|---:|
| Full Fine-tuning | 277,463,053 |
| LoRA | 304,909 |
| Bottleneck Adapter | 1,746,655 |

## 🔎 یافته‌های اصلی بخش PEFT

روش Full Fine-tuning با `F1=0.9507` بهترین عملکرد را روی مجموعه Test داشت؛ زیرا تمام وزن‌های مدل برای مسئله NER فارسی به‌روزرسانی شدند.

Bottleneck Adapter با آموزش تنها `0.6243%` از پارامترهای مدل به `F1=0.8386` رسید و بهترین روش کم‌پارامتر در این آزمایش بود.

LoRA تنها `0.1098%` از پارامترها را آموزش داد و کم‌هزینه‌ترین روش از نظر تعداد پارامترهای قابل‌آموزش بود، اما با تنظیمات فعلی عملکرد پایین‌تری نسبت به Adapter داشت.

Adapter نسبت به LoRA حدود `0.0583` امتیاز F1 بهتر عمل کرد، درحالی‌که همچنان کمتر از یک درصد پارامترهای مدل را آموزش داد.

انتخاب روش مناسب به محدودیت حافظه، زمان آموزش و دقت موردنیاز بستگی دارد:

- بیشترین دقت → Full Fine-tuning
- بهترین تعادل کیفیت و هزینه → Bottleneck Adapter
- کمترین پارامتر قابل‌آموزش → LoRA

## ⚙️ تنظیمات روش‌ها

### Full Fine-tuning

```text
Epochs: 3
Learning rate: 2e-5
Effective batch size: 8
Trainable parameters: 100%
```

### LoRA

```text
r: 8
lora_alpha: 16
lora_dropout: 0.1
Target modules: query, value
Learning rate: 1e-4
Epochs: 3
```

### Bottleneck Adapter

```text
Adapter type: seq_bn
Learning rate: 1e-4
Epochs: 3
Trainable parameters: 0.6243%
```

فایل نتایج نهایی این بخش:

```text
results/final_comparison.csv
```

---

# بخش دوم: انتقال بین‌زبانی با MAD-X

## 🌍 هدف آزمایش MAD-X

هدف این بخش بررسی این موضوع است که آیا می‌توان دانش وظیفه NER را از انگلیسی به فارسی منتقل کرد، بدون اینکه Task Adapter با داده برچسب‌دار فارسی آموزش ببیند.

در این آزمایش:

- Language Adapter فارسی با متن‌های بدون برچسب فارسی آموزش داده شد.
- Language Adapter انگلیسی با متن‌های بدون برچسب انگلیسی آموزش داده شد.
- Task Adapter مربوط به NER فقط با داده‌های برچسب‌دار انگلیسی آموزش داده شد.
- برای ارزیابی فارسی، فقط Language Adapter انگلیسی با Language Adapter فارسی جایگزین شد.
- هیچ داده برچسب‌دار فارسی برای آموزش Task Adapter استفاده نشد.

## 📊 دیتاست WikiANN

منبع:

```text
unimelb-nlp/wikiann
```

پیکربندی‌های استفاده‌شده:

```text
English: en
Persian: fa
```

تقسیم‌بندی هر زبان:

| بخش | تعداد نمونه |
|---|---:|
| Train | 20,000 |
| Validation | 10,000 |
| Test | 10,000 |

برای آموزش هر Language Adapter از ۱۸ هزار جمله بدون برچسب برای Train و ۲ هزار جمله برای Validation استفاده شد.

## 🏷️ برچسب‌های WikiANN

در این بخش از ۷ برچسب BIO استفاده شد:

| ID | برچسب | معنی |
|---:|---|---|
| 0 | `O` | خارج از موجودیت |
| 1 | `B-PER` | شروع موجودیت شخص |
| 2 | `I-PER` | ادامه موجودیت شخص |
| 3 | `B-ORG` | شروع موجودیت سازمان |
| 4 | `I-ORG` | ادامه موجودیت سازمان |
| 5 | `B-LOC` | شروع موجودیت مکان |
| 6 | `I-LOC` | ادامه موجودیت مکان |

## 🧩 معماری MAD-X

### 1. آموزش Language Adapter فارسی

```text
Persian unlabeled text
        ↓
XLM-R + Persian Language Adapter
        ↓
Masked Language Modeling
```

در این مرحله فقط `fa_language` آموزش داده شد.

### 2. آموزش Language Adapter انگلیسی

```text
English unlabeled text
        ↓
XLM-R + English Language Adapter
        ↓
Masked Language Modeling
```

در این مرحله فقط `en_language` آموزش داده شد.

### 3. آموزش NER Task Adapter

```text
English labeled WikiANN
        ↓
English Language Adapter
        ↓
NER Task Adapter
        ↓
NER Head
```

در این مرحله مدل پایه و Language Adapterها ثابت بودند و فقط `ner_task` و NER Head آموزش داده شدند.

### 4. انتقال Zero-shot به فارسی

```text
Persian WikiANN test
        ↓
Persian Language Adapter
        ↓
English-trained NER Task Adapter
        ↓
NER Head
```

در این مرحله Task Adapter بدون تغییر باقی ماند و فقط Language Adapter فارسی فعال شد.

## ⚙️ تنظیمات MAD-X

### Language Adapterهای فارسی و انگلیسی

```text
Adapter config: SeqBnInvConfig
Reduction factor: 2
Non-linearity: relu
Invertible adapter: nice
Epochs: 3
Learning rate: 1e-4
Batch size: 16
Gradient accumulation: 2
Effective batch size: 32
Maximum sequence length: 128
```

### NER Task Adapter

```text
Adapter config: SeqBnConfig
Reduction factor: 16
Epochs: 5
Learning rate: 1e-4
Batch size: 16
Gradient accumulation: 2
Effective batch size: 32
Maximum sequence length: 128
Best model metric: F1
```

پارامترهای قابل‌آموزش Task Adapter و NER Head:

```text
1,742,041 parameters
0.5914% of the model
```

## 📉 نتایج Language Adapterها

### Persian Language Adapter

| معیار | مقدار |
|---|---:|
| Best validation loss | 1.9032 |
| Final evaluation loss | 2.0064 |
| Perplexity | 7.4368 |
| Trainable parameters | 7,979,904 |
| Trainable percentage | 2.7875% |

### English Language Adapter

| معیار | مقدار |
|---|---:|
| Best validation loss | 2.5046 |
| Final evaluation loss | 2.5076 |
| Perplexity | 12.2758 |
| Trainable parameters | 7,979,904 |
| Trainable percentage | 2.7174% |

## 🎯 نتایج نهایی MAD-X

| حالت ارزیابی | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| English source test | 0.7509 | 0.7872 | 0.7686 | 0.9115 |
| Persian zero-shot test | 0.7412 | 0.8093 | **0.7738** | 0.8891 |

Task Adapter فقط با داده برچسب‌دار انگلیسی آموزش داده شد، اما در ارزیابی Zero-shot فارسی به `F1=0.7738` رسید.

Recall فارسی برابر با `0.8093` بود که از Recall انگلیسی بیشتر است؛ یعنی مدل درصد بیشتری از موجودیت‌های فارسی را شناسایی کرده است. در مقابل Precision فارسی کمی کمتر بود، بنابراین تعدادی پیش‌بینی مثبت اشتباه بیشتر ایجاد شده است.

بالاتر بودن جزئی F1 فارسی به این معنا نیست که مدل به‌طور کلی روی فارسی بهتر از انگلیسی است؛ زیرا مجموعه‌های تست دو زبان یکسان نیستند و ممکن است از نظر دشواری، طول جمله و توزیع موجودیت‌ها متفاوت باشند.

## ⚠️ نکته مهم درباره مقایسه نتایج

نتایج ARMAN و WikiANN نباید مستقیماً در یک جدول به‌عنوان رقابت بین روش‌ها تفسیر شوند.

بخش اول:

```text
Dataset: ARMAN
Task: supervised Persian NER
Labels: 13
```

بخش MAD-X:

```text
Dataset: WikiANN
Task: cross-lingual zero-shot NER
Labels: 7
```

به‌دلیل تفاوت دیتاست، نوع برچسب‌ها و تنظیم آزمایش، مقایسه مستقیم اعداد F1 این دو بخش از نظر علمی صحیح نیست.

## ⚠️ محدودیت آزمایش MAD-X

این پیاده‌سازی یک نسخه سبک و آموزشی از MAD-X است.

Language Adapterهای فارسی و انگلیسی فقط با ۱۸ هزار جمله کوتاه WikiANN آموزش داده شدند. در پیاده‌سازی‌های کامل‌تر معمولاً از پیکره‌های تک‌زبانه بسیار بزرگ‌تر و متنوع‌تر استفاده می‌شود.

بنابراین استفاده از داده بیشتر، آموزش طولانی‌تر و تنظیم دقیق‌تر Adapterها می‌تواند کیفیت انتقال بین‌زبانی را بهبود دهد.

---

## 🛠 ابزارها

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT
Adapters
Evaluate
Seqeval
Matplotlib
Google Colab
```

نسخه‌های اصلی استفاده‌شده در بخش MAD-X:

```text
transformers==4.57.6
adapters==1.3.0
datasets==4.0.0
evaluate==0.4.6
seqeval==1.2.2
```

---

## 📁 ساختار پروژه

```text
persian-ner-peft/
│
├── notebooks/
│   ├── README.md
│   ├── Persian_NER_PEFT_Project_Final.ipynb
│   └── Persian_NER_MADX_Extension.ipynb
│
├── results/
│   ├── final_comparison.csv
│   ├── madx_comparison.csv
│   ├── madx_zero_shot_results.json
│   ├── fa_language_adapter_results.json
│   └── en_language_adapter_results.json
│
├── README.md
└── LICENSE
```

مدل‌ها، Adapterها و checkpointهای حجیم به‌دلیل محدودیت حجم در GitHub قرار نگرفته‌اند و در Google Drive ذخیره شده‌اند.

خروجی‌های اصلی MAD-X در Google Drive:

```text
outputs/fa_language_adapter
outputs/en_language_adapter
outputs/en_ner_task_adapter
```

---

## 🚀 اجرای پروژه

### اجرای بخش PEFT

نوت‌بوک زیر را در Google Colab باز کنید:

```text
notebooks/Persian_NER_PEFT_Project_Final.ipynb
```

GPU را فعال کنید:

```text
Runtime → Change runtime type → T4 GPU
```

بخش‌های Full Fine-tuning و LoRA با محیط اولیه اجرا شده‌اند.

برای اجرای Bottleneck Adapter، نسخه‌های سازگار زیر نصب شدند:

```text
transformers==4.57.6
adapters==1.3.0
```

پس از نصب، Colab Session باید یک بار Restart شود. بنابراین بخش Adapter باید پس از Restart ادامه داده شود.

### اجرای بخش MAD-X

نوت‌بوک زیر را باز کنید:

```text
notebooks/Persian_NER_MADX_Extension.ipynb
```

سپس:

1. GPU مدل T4 را فعال کنید.
2. سلول نصب کتابخانه‌ها را اجرا کنید.
3. از مسیر زیر Session را Restart کنید:

```text
Runtime → Restart session
```

4. اجرای نوت‌بوک را از سلول بررسی نسخه کتابخانه‌ها ادامه دهید.
5. سلول نصب را پس از Restart دوباره اجرا نکنید.

اجرای کامل MAD-X شامل آموزش دو Language Adapter و یک NER Task Adapter است و روی Tesla T4 حدود ۴۰ دقیقه زمان می‌برد.

---

## 📈 فایل‌های نتایج

### نتایج بخش PEFT

```text
results/final_comparison.csv
```

### نتایج بخش MAD-X

```text
results/fa_language_adapter_results.json
results/en_language_adapter_results.json
results/madx_zero_shot_results.json
results/madx_comparison.csv
```

---

## 🔭 مراحل بعدی

- تنظیم دقیق‌تر پارامترهای `r`، `alpha` و dropout در LoRA
- بررسی لایه‌های هدف بیشتر برای LoRA
- استفاده از Early Stopping
- تحلیل خطا برای هر نوع موجودیت
- بررسی عملکرد جداگانه روی `PER`، `ORG` و `LOC`
- آموزش Language Adapterها با پیکره‌های بزرگ‌تر تک‌زبانه
- آزمایش انتقال از زبان‌های مبدأ دیگر
- مقایسه MAD-X با Zero-shot مستقیم بدون Language Adapter
- استقرار مدل به‌صورت API یا برنامه تحت وب

---

## 👩‍💻 توسعه‌دهنده

Hannaneh Sepehri — MSc AI Student
