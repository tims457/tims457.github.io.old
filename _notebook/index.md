---
layout: page
title: Notebook
description: Tim's notes
permalink: /notebook/
---

<ul>
{% assign sortedNotebook = site.notebook | sort: "date" | reverse %}

{% for post in sortedNotebook %}
    {% if post.title != 'Notebook' %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » {% if post.highlight %}&starf; {% endif %}<a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
    {% endif %}
{% endfor %}

</ul>