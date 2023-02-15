---
layout: page
title: Notebook
description: Tim's notes
permalink: /notebook/
---

<ul>
  {% for post in site.categories.notebook %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » {% if post.highlight %}&starf; {% endif %}<a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
