
نه اون مقاله برای این درس نیست. الان فایل ها رو میدم. صبر کن


6-section6-parameter efficient fine tuning.pdf
PDF

7- section7-parameter efficient fine-tuning (II).pdf
PDF

4-section4-Transformers (part3).pdf
PDF

5-section5-zero-shot learning.pdf
PDF

2-section2-introduction to Transformers.pdf
PDF

3-section3-Transformers (part2)_1.pdf
PDF

1-section1-introduction.pdf
PDF


Hardware accelerator → GPU

این و پیدا نمیکنم. این گزینه ها رو داره


Sunday 2:18 PM
الان داره درست انجام میشه ؟

Sunday 4:08 PM
وسطش وایساد و دوباره ریکانکت رو زدم. از اول باید اجرا شن ؟

Worked for 38s
اگر عدد 3678/6915 بیشتر از حدود ۳ تا ۵ دقیقه ثابت مانده، نه؛ احتمالاً آموزش دیگر در حال اجرا نیست. وصل‌بودن Runtime به‌تنهایی کافی نیست.

به علامت سمت چپ همان سلول نگاه کن:

اگر مربع توقف ⏹ می‌بینی، سلول هنوز در حال اجراست؛ کمی صبر کن.

اگر مثلث اجرا ▶ می‌بینی، آموزش متوقف شده است.

نگران نباش؛ پایان epoch اول ذخیره شده و در صورت توقف، می‌توانیم از همان checkpoint ادامه بدهیم، نه از اول. فقط بگو کنار سلول مربع می‌بینی یا مثلث.



تو این مرحله بودیم.gpu هم الان دارم. درست ادامه بده

Monday 5:24 AM

رو 12 درصد گیر کرده. استپش کنم از اول اجرا کنم ؟ یا طبیعیه خیلی طول بکشه ؟

Wednesday 7:14 AM

Persian_NER_PEFT_Project.ipynb
File
این کدی هست که تا الان زدم. الان دوباره gpu دارم .
بهم بگو از اول کدوم سلول ها رو اجرا کنم ؟

مربع نداره داره میچرخه مثلثش

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.0/44.0 kB 2.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 295.5/295.5 kB 13.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 101.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 566.4/566.4 kB 50.8 MB/s eta 0:00:00
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio 6.19.0 requires huggingface-hub<2.0,>=1.2.0, but you have huggingface-hub 0.36.2 which is incompatible.


Persian_NER_PEFT_Project (1).ipynb
File
من قاطی کردم و بلد نیستم چیکار کنم. الان بهت فایل رو میدم. تکه تکه بگو چیکار کنم


Copy_of_Persian_NER_PEFT_Project_Final.ipynb
File


Copy_of_Persian_NER_PEFT_Project_Final (1).ipynb
File

نمودار اجرا و فایل ذخیره شد.


Copy_of_Persian_NER_PEFT_Project_Final (2).ipynb
File


Persian_NER_PEFT_Presentation_FA(1).pptx
Presentation
هر جور میخوای اصلاحش کن . فقط اون اوایلش 13 تا برچسب رو هم بگو که شامل چیا هستن دقیق

اینجوریه . الان دیگه نوت بوک ندارم

پوشه notebooks ساخته شد ادامه بده

 [125/125 00:27]
Language Adapter فارسی ذخیره شد.
مسیر Adapter: /content/drive/MyDrive/persian-ner-madx/outputs/fa_language_adapter
Validation Loss: 2.0064
Perplexity: 7.4368
مسیر نتایج: /content/drive/MyDrive/persian-ner-madx/results/fa_language_adapter_results.json


Google Drive storage quota has been exceeded (most recently at 11:53:29 AM). این مشکل رو هم دارم . الان خیلی به درایوم نیاز داریم ؟ یا دیگه چیزی واسه ذخیره نیست ؟


یک شرکت دارای یک مدل مبتنی بر ترانسفورمر از پیش آموزش داده شده است. این شرکت دارای مشتریان متعددی است که می‌خواهند برایشان تسک خلاصه سازی متن در زمینه های مختلف (متون پزشکی، حقوقی، مالی و اخبار) انجام شود‌.
محدودیت های سخت افزاری و منابع جهت نگهداری یک مدل منفرد برای هر مشتری وجود دارد. همچنین مدل پایه باید بدون تغییر باقی بماند. یک روش انطباق(adaptation) مناسب برای این شرکت پیشنهاد دهید. ‌ نکته بسیار مهم توضیح علت پیشنهاد روش مورد استفاده است.

چیکار کنم


Copy_of_Persian_NER_MADX_Extension.ipynb
File
بگو بعد از کتابخانه ها کدام سلول ها اجرا بشن یا کدوما نشن. من قاطی میکنم 

عالی؛ آموزش درست انجام شده و Best F1 = 0.7677 تقریباً با اجرای قبلی برابر است. هشدارهای Overwriting existing adapter/head خطا نیستند؛ هنگام برگرداندن بهترین checkpoint، Trainer نسخه‌ی ذخیره‌شده‌ی Adapterها و Head را دوباره روی مدل بارگذاری کرده است.

الان مهم‌ترین کار این است که قبل از قطع‌شدن Runtime، Task Adapter و NER Head را روی Drive ذخیره کنیم. این سلول را اجرا کن:

# این سلول Task Adapter آموزش‌دیده و NER Head را روی Google Drive ذخیره می‌کند.

from pathlib import Path
import shutil

TASK_ADAPTER_DIR = Path(
    "/content/drive/MyDrive/persian-ner-madx/outputs/en_ner_task_adapter"
)

# پوشه خالی یا ناقص قبلی پاک می‌شود.
if TASK_ADAPTER_DIR.exists():
    shutil.rmtree(TASK_ADAPTER_DIR)

TASK_ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)

task_model.save_adapter(
    str(TASK_ADAPTER_DIR),
    "ner_task",
    with_head=True
)

print("Task Adapter ذخیره شد:\n")

for file in TASK_ADAPTER_DIR.rglob("*"):
    if file.is_file():
        size_mb = file.stat().st_size / (1024 ** 2)
        print(f"- {file.relative_to(TASK_ADAPTER_DIR)} | {size_mb:.2f} MB")
خروجی این سلول را بفرست. باید علاوه بر فایل Adapter، فایل‌های مربوط به Head با نام ner_task هم دیده شوند.


Task Adapter ذخیره شد:

- adapter_config.json | 0.00 MB
- pytorch_adapter.bin | 3.43 MB
- head_config.json | 0.00 MB
- pytorch_model_head.bin | 0.02 MB

Worked for 1m 2s
خیلی خوب؛ Task Adapter و Head کامل ذخیره شده‌اند. حالا ارزیابی نهایی را انجام می‌دهیم:

ابتدا روی تست انگلیسی با Language Adapter انگلیسی

سپس روی تست فارسی به‌صورت Zero-shot با جایگزین‌کردن Language Adapter فارسی

در پایان، نتایج روی Drive ذخیره می‌شوند

این سلول را اجرا کن:

# این سلول مدل را روی تست انگلیسی و سپس به‌صورت Zero-shot روی تست فارسی
# ارزیابی می‌کند و نتایج نهایی را در Google Drive ذخیره می‌کند.

import json
from adapters import Stack

def select_metrics(metrics):
    wanted = ["eval_precision", "eval_recall", "eval_f1", "eval_accuracy"]
    return {
        key.replace("eval_", ""): float(metrics[key])
        for key in wanted
    }


# ارزیابی روی زبان مبدأ: انگلیسی
task_model.set_active_adapters(
    Stack(EN_LANG_ADAPTER_NAME, TASK_ADAPTER_NAME)
)
task_model.active_head = TASK_ADAPTER_NAME

english_test_metrics = ner_trainer.evaluate(
    en_ner_tokenized["test"]
)

english_results = select_metrics(english_test_metrics)


# ارزیابی Zero-shot روی زبان هدف: فارسی
task_model.set_active_adapters(
    Stack(LANG_ADAPTER_NAME, TASK_ADAPTER_NAME)
)
task_model.active_head = TASK_ADAPTER_NAME

persian_test_metrics = ner_trainer.evaluate(
    fa_ner_tokenized["test"]
)

persian_results = select_metrics(persian_test_metrics)


# ذخیره نتایج نهایی
madx_results = {
    "english_source_test": english_results,
    "persian_zero_shot_test": persian_results,
    "best_english_validation_f1": float(ner_trainer.state.best_metric)
}

result_path = RESULTS_DIR / "madx_zero_shot_results.json"

with open(result_path, "w", encoding="utf-8") as file:
    json.dump(madx_results, file, ensure_ascii=False, indent=2)


print("\nنتایج تست انگلیسی:")
for name, value in english_results.items():
    print(f"{name}: {value:.4f}")

print("\nنتایج Zero-shot فارسی:")
for name, value in persian_results.items():
    print(f"{name}: {value:.4f}")

print("\nنتایج ذخیره شد در:")
print(result_path)
اجرای این سلول ممکن است چند دقیقه طول بکشد. خروجی هر دو بخش انگلیسی و فارسی را بفرست.


نتایج تست انگلیسی:
precision: 0.7509
recall: 0.7872
f1: 0.7686
accuracy: 0.9115

نتایج Zero-shot فارسی:
precision: 0.7412
recall: 0.8093
f1: 0.7738
accuracy: 0.8891

نتایج ذخیره شد در:
/content/drive/MyDrive/persian-ner-madx/results/madx_zero_shot_results.json

نتیجه خیلی خوب است. بخش اصلی MAD-X با موفقیت کامل شد:

انگلیسی: F1 = 0.7686

فارسی Zero-shot: F1 = 0.7738

یعنی Task Adapter فقط با داده برچسب‌دار انگلیسی آموزش دیده، اما با قرارگرفتن کنار Persian Language Adapter توانسته روی فارسی هم بدون داده برچسب‌دار فارسی عملکرد مناسبی داشته باشد. کمی بالاتر بودن F1 فارسی جالب است، ولی چون مجموعه‌های تست انگلیسی و فارسی دقیقاً نمونه‌های یکسانی نیستند، نباید نتیجه بگیریم مدل روی فارسی ذاتاً بهتر است.

حالا یک جدول نهایی جداگانه برای نتایج MAD-X ذخیره می‌کنیم. این سلول را اجرا کن:

# این سلول نتایج انگلیسی و فارسی Zero-shot را در یک فایل CSV جدا ذخیره می‌کند.

import pandas as pd

comparison_df = pd.DataFrame([
    {
        "setting": "English source test",
        "language_adapter": "English",
        "task_training_language": "English",
        "precision": english_results["precision"],
        "recall": english_results["recall"],
        "f1": english_results["f1"],
        "accuracy": english_results["accuracy"]
    },
    {
        "setting": "Persian zero-shot test",
        "language_adapter": "Persian",
        "task_training_language": "English",
        "precision": persian_results["precision"],
        "recall": persian_results["recall"],
        "f1": persian_results["f1"],
        "accuracy": persian_results["accuracy"]
    }
])

csv_path = RESULTS_DIR / "madx_comparison.csv"
comparison_df.to_csv(csv_path, index=False)

display(comparison_df)

print("\nفایل جدول ذخیره شد:")
print(csv_path)
این جدول را فعلاً با نتایج پروژه اصلی ARMAN ترکیب نکن؛ چون نتایج قبلی روی دیتاست ARMAN و این نتایج روی WikiANN محاسبه شده‌اند و مقایسه مستقیم آن‌ها علمی نیست. خروجی جدول را بفرست تا بخش تحلیل و جمع‌بندی نهایی نوت‌بوک را اضافه کنیم.


setting	language_adapter	task_training_language	precision	recall	f1	accuracy
0	English source test	English	English	0.750910	0.787174	0.768614	0.911498
1	Persian zero-shot test	Persian	English	0.741176	0.809346	0.773763	0.889109



فایل جدول ذخیره شد:
/content/drive/MyDrive/persian-ner-madx/results/madx_comparison.csv

عالی. حالا بخش تحلیل نتایج MAD-X را به نوت‌بوک اضافه می‌کنیم.

یک Text/Markdown cell زیر جدول بساز و این متن را عیناً داخلش قرار بده:

## تحلیل نتایج MAD-X

در این آزمایش، Task Adapter مربوط به تشخیص موجودیت نامدار فقط با داده‌های برچسب‌دار انگلیسی WikiANN آموزش داده شد. سپس برای ارزیابی انگلیسی، ترکیب زیر استفاده شد:

English Language Adapter → NER Task Adapter

برای ارزیابی Zero-shot فارسی، بدون آموزش Task Adapter با هیچ داده برچسب‌دار فارسی، فقط Language Adapter انگلیسی با Language Adapter فارسی جایگزین شد:

Persian Language Adapter → English-trained NER Task Adapter

نتایج نشان دادند که مدل روی داده تست انگلیسی به F1 برابر با 0.7686 و روی داده تست فارسی در حالت Zero-shot به F1 برابر با 0.7738 رسید. بنابراین Task Adapter توانسته دانش وظیفه NER را که از زبان انگلیسی یاد گرفته بود، با استفاده از Language Adapter فارسی به زبان هدف منتقل کند.

Recall مدل روی فارسی برابر با 0.8093 است که از Recall انگلیسی بیشتر است. این موضوع نشان می‌دهد مدل درصد بیشتری از موجودیت‌های فارسی را پیدا کرده است. بااین‌حال Precision فارسی کمی کمتر است؛ یعنی مدل در زبان فارسی تعدادی پیش‌بینی مثبت اشتباه بیشتری داشته است.

با وجود بالاتر بودن جزئی F1 فارسی، نمی‌توان نتیجه گرفت که مدل ذاتاً روی فارسی بهتر از انگلیسی عمل می‌کند؛ زیرا مجموعه‌های تست فارسی و انگلیسی شامل نمونه‌های یکسان نیستند و ممکن است از نظر دشواری، طول جمله و توزیع موجودیت‌ها متفاوت باشند.

این نتایج را نباید مستقیماً با نتایج Full Fine-tuning، LoRA و Adapter پروژه اصلی روی دیتاست ARMAN مقایسه کرد؛ زیرا آزمایش اصلی روی ARMAN و آزمایش MAD-X روی WikiANN انجام شده‌اند و نوع برچسب‌ها و توزیع داده‌های آن‌ها متفاوت است.

### محدودیت آزمایش

این پیاده‌سازی یک نسخه سبک و آموزشی از MAD-X است. Language Adapterهای فارسی و انگلیسی فقط با ۱۸ هزار جمله کوتاه WikiANN آموزش داده شدند، درحالی‌که در نسخه اصلی MAD-X معمولاً از حجم بسیار بیشتری از متن‌های تک‌زبانه، مانند داده‌های Wikipedia، برای آموزش Language Adapter استفاده می‌شود. بنابراین استفاده از داده بیشتر و متنوع‌تر می‌تواند کیفیت انتقال بین‌زبانی را بهتر کند.

### نتیجه‌گیری

نتایج نشان می‌دهند که جداسازی دانش زبان و دانش وظیفه با استفاده از Language Adapter و Task Adapter، امکان اجرای NER فارسی به‌صورت Cross-lingual Zero-shot را فراهم کرده است. در این روش هیچ داده برچسب‌دار فارسی برای آموزش Task Adapter استفاده نشد و مدل پایه XLM-R نیز ثابت باقی ماند.
بعد از اضافه‌کردن این متن، فقط بنویس «اضافه شد» تا بخش جمع‌بندی معماری و فایل‌های خروجی نوت‌بوک را کامل کنیم.


اضافه شد

خیلی خوب؛ محتوای نوت‌بوک کامل شد. حالا نسخه نهایی را ذخیره کنیم:

بالای Colab روی نام فایل بزن و نامش را دقیقاً به این تغییر بده:

Persian_NER_MADX_Extension.ipynb
صبر کن تا کنار نام فایل عبارت Saved to Drive نمایش داده شود.

از منوی زیر نسخه نهایی را دانلود کن:

File → Download → Download .ipynb
فایل دانلودشده را همین‌جا بفرست تا ترتیب سلول‌ها، کدهای تکراری، خروجی‌ها و متن‌های نهایی را یک‌بار کامل بررسی کنم و بعد برای GitHub آماده‌اش کنیم.



Persian_NER_MADX_Extension.ipynb
File

این فایل را اول دانلود کن:

دانلود نسخه آماده GitHub

بعد نام فایل دانلودشده را به این تغییر بده:

Persian_NER_MADX_Extension.ipynb
حالا در GitHub این مراحل را انجام بده:

وارد مخزن پروژه شو:

HannanehSepehri/persian-ner-peft
پوشه‌ی notebooks را باز کن.

بالای صفحه روی Add file بزن.

گزینه‌ی Upload files را انتخاب کن.

فایل زیر را داخل صفحه بکش یا با choose your files انتخاب کن:

Persian_NER_MADX_Extension.ipynb
پایین صفحه، در قسمت Commit message بنویس:

Add MAD-X cross-lingual zero-shot NER extension
گزینه‌ی Commit directly to the main branch را انتخاب کن.

روی Commit changes بزن.

نکته مهم: این فایل را به‌عنوان نوت‌بوک جدید داخل notebooks بگذار و فایل پروژه اصلی یعنی:

Persian_NER_PEFT_Project_Final.ipynb
را حذف یا جایگزین نکن. در نهایت داخل پوشه notebooks باید هر دو فایل دیده شوند.


حالا readme جدید رو بده که این توضیحاتم داشته باشه. اگه میخوای قبلی رو بدم

README قبلی را بفرست تا همان ساختار و توضیحات پروژه اصلی Full Fine-tuning، LoRA و Adapter حفظ شود و بخش جدید MAD-X و Zero-shot فارسی را دقیق و بدون حذف مطالب قبلی به آن اضافه کنم.

فایل README.md را آپلود کن یا متن کاملش را همین‌جا بفرست.


# 🔍 Persian Named Entity Recognition with PEFT

تشخیص موجودیت‌های نامدار فارسی و مقایسه روش‌های **Full Fine-tuning، LoRA و Bottleneck Adapter**

## 📌 درباره پروژه

این پروژه با استفاده از مدل چندزبانه XLM-RoBERTa-base، موجودیت‌های نامدار موجود در متن‌های فارسی را شناسایی می‌کند.

هدف اصلی پروژه، مقایسه آموزش کامل مدل با روش‌های کم‌پارامتر است تا مشخص شود با آموزش درصد بسیار کمی از پارامترهای مدل، چه مقدار از کیفیت تشخیص موجودیت‌ها حفظ می‌شود.

سه روش زیر آموزش داده و مقایسه شدند:

- Full Fine-tuning
- LoRA
- Bottleneck Adapter

## 📊 دیتاست

منبع: Hugging Face — hezarai/arman-ner

دیتاست ARMAN یکی از دیتاست‌های فارسی تشخیص موجودیت نامدار است. هر نمونه شامل کلمات جمله و برچسب NER مربوط به هر کلمه است.

تقسیم‌بندی نهایی داده‌ها:

| بخش | تعداد نمونه |
|---|---:|
| Train | 18,435 |
| Validation | 2,049 |
| Test | 2,561 |

مجموعه آموزشی با seed=42 به بخش‌های Train و Validation تقسیم شد و مجموعه Test تا پایان آموزش بدون تغییر باقی ماند.

## 🏷️ برچسب‌های پروژه

در این پروژه از قالب BIO و در مجموع **۱۳ برچسب** استفاده شده است:

| ID | برچسب | معنی |
|---:|---|---|
| 0 | O | خارج از هر موجودیت |
| 1 | B-pro | شروع موجودیت محصول |
| 2 | I-pro | ادامه موجودیت محصول |
| 3 | B-pers | شروع موجودیت شخص |
| 4 | I-pers | ادامه موجودیت شخص |
| 5 | B-org | شروع موجودیت سازمان |
| 6 | I-org | ادامه موجودیت سازمان |
| 7 | B-loc | شروع موجودیت مکان |
| 8 | I-loc | ادامه موجودیت مکان |
| 9 | B-fac | شروع موجودیت تأسیسات |
| 10 | I-fac | ادامه موجودیت تأسیسات |
| 11 | B-event | شروع موجودیت رویداد |
| 12 | I-event | ادامه موجودیت رویداد |

شش نوع موجودیت موجود در دیتاست:

- pers — شخص، مانند «علی دایی»
- org — سازمان، مانند «سازمان ملل»
- loc — مکان، مانند «ایران»
- fac — تأسیسات یا بنا، مانند «برج میلاد»
- pro — محصول، مانند «آیفون»
- event — رویداد، مانند «جام جهانی»

در قالب BIO:

- B نشان‌دهنده شروع یک موجودیت است.
- I نشان‌دهنده ادامه همان موجودیت است.
- O نشان‌دهنده کلمه‌ای خارج از موجودیت است.

## 🛠 ابزارها

Python | PyTorch | Hugging Face Transformers | Datasets | PEFT | Adapters | Evaluate | Seqeval | Matplotlib | Google Colab

مدل پایه:

text
xlm-roberta-base


محیط اجرا:

text
Google Colab — Tesla T4 GPU


## 🔧 آماده‌سازی داده

توکنایزر XLM-RoBERTa ممکن است یک کلمه فارسی را به چند زیرتوکن تقسیم کند.

برای هماهنگ‌کردن برچسب‌ها با زیرتوکن‌ها:

- اولین زیرتوکن، برچسب اصلی کلمه را دریافت کرد.
- زیرتوکن‌های بعدی مقدار -100 گرفتند.
- توکن‌های ویژه نیز مقدار -100 گرفتند.
- مقادیر -100 در محاسبه Loss و معیارهای ارزیابی نادیده گرفته شدند.

حداکثر طول ورودی:

text
256 tokens


## 🤖 مدل‌سازی و نتایج

سه روش آموزش داده و مقایسه شدند:

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

## 🔎 یافته‌های کلیدی مدل‌سازی

روش Full Fine-tuning با F1=0.9507 بهترین عملکرد را روی مجموعه Test داشت؛ زیرا تمام وزن‌های مدل برای مسئله NER فارسی به‌روزرسانی شدند.

Bottleneck Adapter با آموزش تنها 0.6243% از پارامترهای مدل به F1=0.8386 رسید و بهترین روش کم‌پارامتر در این آزمایش بود.

LoRA تنها 0.1098% از پارامترها را آموزش داد و کم‌هزینه‌ترین روش از نظر تعداد پارامترهای قابل‌آموزش بود، اما با تنظیمات فعلی عملکرد پایین‌تری نسبت به Adapter داشت.

Adapter نسبت به LoRA حدود 0.0583 امتیاز F1 بهتر عمل کرد، درحالی‌که همچنان کمتر از یک درصد پارامترهای مدل را آموزش داد.

کاهش شدید تعداد پارامترهای قابل‌آموزش لزوماً بدون افت کیفیت نیست و انتخاب روش مناسب به محدودیت حافظه، زمان آموزش و دقت موردنیاز بستگی دارد.

انتخاب روش نهایی:

- بیشترین دقت → Full Fine-tuning
- بهترین تعادل کیفیت و هزینه → Bottleneck Adapter
- کمترین پارامتر قابل‌آموزش → LoRA

## ⚙️ تنظیمات روش‌ها

### Full Fine-tuning

text
Epochs: 3
Learning rate: 2e-5
Effective batch size: 8
Trainable parameters: 100%


### LoRA

text
r: 8
lora_alpha: 16
lora_dropout: 0.1
Target modules: query, value
Learning rate: 1e-4
Epochs: 3


### Bottleneck Adapter

text
Adapter type: seq_bn
Learning rate: 1e-4
Epochs: 3
Trainable parameters: 0.6243%


## 📈 نمودارها

نمودار مقایسه F1 سه روش در نوت‌بوک نهایی قرار دارد:

- Full Fine-tuning: 0.9507
- Bottleneck Adapter: 0.8386
- LoRA: 0.7803

فایل نتایج نهایی:

text
results/final_comparison.csv


## 📁 ساختار پروژه

text
persian-ner-peft/
│
├── notebooks/
│   ├── README.md
│   └── Persian_NER_PEFT_Project_Final.ipynb
│
├── results/
│   └── final_comparison.csv
│
├── README.md
└── LICENSE


## 🚀 اجرا

نوت‌بوک زیر را در Google Colab باز کنید:

text
notebooks/Persian_NER_PEFT_Project_Final.ipynb


سپس GPU را فعال کنید:

text
Runtime → Change runtime type → T4 GPU


### نکته مهم

بخش‌های Full Fine-tuning و LoRA با محیط اولیه اجرا شده‌اند.

برای اجرای Bottleneck Adapter، نسخه‌های سازگار زیر نصب شدند:

text
transformers==4.57.6
adapters==1.3.0


پس از نصب این نسخه‌ها، Colab Session باید یک بار Restart شود.

بنابراین نوت‌بوک نباید از ابتدا تا انتها با گزینه Run all اجرا شود و بخش Adapter باید پس از Restart ادامه داده شود.

مدل‌ها و checkpointهای حجیم به‌دلیل محدودیت حجم در GitHub قرار نگرفته‌اند و در Google Drive ذخیره شده‌اند.

## 🔭 مراحل بعدی

- تنظیم دقیق‌تر پارامترهای r، alpha و dropout در LoRA
- بررسی لایه‌های هدف بیشتر برای LoRA
- افزایش تعداد epochها و استفاده از Early Stopping
- تحلیل خطا روی موجودیت‌های اشتباه
- مقایسه عملکرد مدل برای هر نوع موجودیت
- افزودن روش MAD-X برای انتقال بین‌زبانی
- استقرار مدل به‌صورت API یا برنامه تحت وب

## 👩‍💻 توسعه‌دهنده

**Hannaneh Sepehri — MSc AI Student**

README جدید آماده شد و تمام توضیحات قبلی به‌همراه بخش کامل MAD-X، معماری، تنظیمات، نتایج Zero-shot، محدودیت‌ها و ساختار جدید پروژه داخل آن قرار گرفته است.

README.md

برای جایگزینی در GitHub:

وارد صفحه اصلی مخزن persian-ner-peft شو.

فایل اصلی README.md را باز کن.

روی آیکون مداد Edit this file بزن.

تمام متن قبلی را پاک کن.

محتوای فایل جدید را جایگزین کن؛ یا از Add file → Upload files همین فایل را آپلود و جایگزینی را تأیید کن.

در Commit message بنویس:

Update README with MAD-X zero-shot extension
روی Commit changes بزن.

در README جدید، عبارت «افزودن MAD-X» از مراحل آینده حذف شده، چون این بخش حالا انجام شده است. همچنین تأکید شده که نتایج ARMAN و WikiANN به‌دلیل تفاوت دیتاست و برچسب‌ها نباید مستقیماً با هم مقایسه شوند. 


Library
/
Hanna
/
README.md


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

**Hannaneh Sepehri — MSc AI Student**
Library
/
Hanna
/
README.md


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
