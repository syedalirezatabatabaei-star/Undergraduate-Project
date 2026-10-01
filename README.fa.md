<div dir="rtl">

<p align="center">
  <a href="README.md">English</a>
  &nbsp; | &nbsp;
  <a href="README.fa.md">فارسی</a>
</p>
</p>
## Technologies

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://skillicons.dev/icons?i=python" width="50" alt="Python">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://skillicons.dev/icons?i=fastapi" width="50" alt="FastAPI">
  </a>
   <a href="https://pytorch.org/">
    <img src="https://skillicons.dev/icons?i=pytorch" width="50" alt="PyTorch">
  </a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/HTML"> 
    <img src="https://skillicons.dev/icons?i=html" width="50" alt="HTML"> 
  </a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/CSS">
    <img src="https://skillicons.dev/icons?i=css" width="50" alt="CSS">
  </a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript">
    <img src="https://skillicons.dev/icons?i=javascript" width="50" alt="JavaScript"> 
  </a>
</p>
  
<p align="center">
  <code>TorchVision</code>
  <code>CNN</code>
  <code>MNIST</code>
  <code>PyTorch Lightning</code>
</p>


# تشخیص اعداد دست‌نویس

پروژه کارشناسی برای تشخیص اعداد دست‌نویس با استفاده از دیتاست MNIST.

این پروژه شامل بخش‌های پیش‌پردازش تصویر، بارگذاری مدل، پیش‌بینی، API و رابط کاربری برای استفاده از مدل‌های آموزش‌دیده است.

## دیتاست

در این پروژه از دیتاست MNIST برای تشخیص اعداد دست‌نویس استفاده شده است.

* ۱۰ کلاس از ۰ تا ۹
* تصاویر Grayscale
* ابعاد تصاویر: ۲۸ × ۲۸ پیکسل

## ساختار پروژه

```text
Undergraduate-Project/
│
├── Frontend/
├── model/
├── models/
│
├── api.py
├── preprocessing.py
├── load.py
├── ensamble.py
│
├── README.md
└── README.fa.md
```

## فایل‌های اصلی

### `api.py`

API مورد استفاده برای دریافت ورودی و برگرداندن نتیجه پیش‌بینی مدل.

### `preprocessing.py`

شامل مراحل پیش‌پردازش موردنیاز قبل از ارسال تصویر به مدل.

### `load.py`

مسئول بارگذاری داده‌ها و منابع موردنیاز مدل.

### `ensamble.py`

شامل بخش مربوط به پیاده‌سازی Ensemble در پروژه.

### `model/`

شامل کدها و اجزای مربوط به مدل.

### `models/`

شامل فایل‌های مدل و منابع مرتبط.

### `Frontend/`

شامل بخش رابط کاربری پروژه.

## روند کلی

روند کلی پیش‌بینی به صورت زیر است:

```text
تصویر ورودی
     |
     v
پیش‌پردازش
     |
     v
مدل آموزش‌دیده
     |
     v
پیش‌بینی
     |
     v
عدد ۰ تا ۹
```

## فناوری‌های استفاده‌شده

* Python
* Machine Learning
* Deep Learning
* Computer Vision
* MNIST
* API
* Frontend

## نصب

ابتدا مخزن را دریافت کنید:

```bash
git clone https://github.com/syedalirezatabatabaei-star/Undergraduate-Project.git
cd Undergraduate-Project
```

سپس وابستگی‌های موردنیاز پروژه را نصب کنید.

در صورت وجود فایل `requirements.txt`:

```bash
pip install -r requirements.txt
```

## نحوه استفاده

پروژه امکان دریافت تصویر عدد دست‌نویس، انجام پیش‌پردازش و ارسال آن به مدل آموزش‌دیده برای تشخیص عدد را فراهم می‌کند.

API به عنوان واسط بین رابط کاربری و سیستم پیش‌بینی مورد استفاده قرار می‌گیرد.

## اطلاعات پروژه

**عنوان:** پروژه کارشناسی — تشخیص اعداد دست‌نویس

**دانشجو:** سید علیرضا طباطبائی

**استاد راهنما:** دکتر ساناز اسدی‌نیا

**استاد مشاور:** دکتر حمیدرضا صدرارحامی

## توسعه‌های آینده

* بهبود عملکرد مدل
* مقایسه معماری‌های مختلف
* اضافه کردن میزان اطمینان پیش‌بینی
* بهبود رابط کاربری
* اضافه کردن تست‌های خودکار
* مستندسازی API
* اجرای پروژه با Docker
* استقرار پروژه

## مجوز

این پروژه با هدف دانشگاهی توسعه داده شده است.

</div>
