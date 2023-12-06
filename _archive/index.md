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
        <div style="float:full"><a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a><span style="float:right">{{ post.date | date_to_string }}</span></div>
    </li>
    {% endif %}
{% endfor %}

</ul>