---
layout: page
title: Archive
permalink: /archive/
---

<ul>
{% assign sortedArchive = site.archive | sort: "date" | reverse %}

{% for post in sortedArchive %}
    {% if post.title != 'Archive' %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » {% if post.highlight %}&starf; {% endif %}<a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
    {% endif %}
{% endfor %}

</ul>